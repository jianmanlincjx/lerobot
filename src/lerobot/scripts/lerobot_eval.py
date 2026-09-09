#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Requires: pip install 'lerobot[evaluation]' plus the policy extra (e.g. lerobot[pi])
          and the environment extra (e.g. lerobot[pusht]) if evaluating in simulation.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
import json
import logging
import os
import pathlib
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import (
    check_env_attributes_and_types,
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    init_logging,
    inside_slurm,
)


def _goal_pose_quantiles(
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the exact goal-pose q01/q99 used by the loaded policy processor."""
    policy_config = getattr(policy, "config", None)
    goal_key = str(getattr(policy_config, "goal_pose_feature_key", "observation.state"))
    stats_candidates: list[Any] = []
    if preprocessor is not None:
        for step in getattr(preprocessor, "steps", []):
            stats_candidates.append(getattr(step, "stats", None))
    stats_candidates.append(getattr(policy_config, "dataset_stats", None))

    for stats in stats_candidates:
        if not isinstance(stats, dict) or not isinstance(stats.get(goal_key), dict):
            continue
        feature_stats = stats[goal_key]
        q01_value = feature_stats.get("q01")
        q99_value = feature_stats.get("q99")
        if q01_value is None or q99_value is None:
            continue
        if torch.is_tensor(q01_value):
            q01_value = q01_value.detach().cpu().numpy()
        if torch.is_tensor(q99_value):
            q99_value = q99_value.detach().cpu().numpy()
        q01 = np.asarray(q01_value, dtype=np.float64).reshape(-1)
        q99 = np.asarray(q99_value, dtype=np.float64).reshape(-1)
        if (
            q01.shape == q99.shape
            and q01.size >= 6
            and np.isfinite(q01).all()
            and np.isfinite(q99).all()
            and np.all(q99[:6] > q01[:6])
        ):
            return q01, q99

    raise RuntimeError(
        f"Cannot visualize normalized goal pose without saved q01/q99 stats for {goal_key!r}."
    )


def _pose_viz_task_prompt(env_i: Any) -> str:
    """Best-effort language instruction from a LIBERO / LIBERO-PRO sub-env."""
    for obj in (env_i, getattr(env_i, "_env", None), getattr(getattr(env_i, "_env", None), "env", None)):
        if obj is None:
            continue
        for key in ("task_description", "language", "task"):
            value = getattr(obj, key, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _draw_prompt_on_frame(
    image: np.ndarray,
    prompt: str,
    *,
    max_width_frac: float = 0.98,
    font_scale: float = 0.42,
) -> None:
    """Draw a wrapped language prompt banner at the bottom of a camera frame."""
    import cv2

    prompt = (prompt or "").strip()
    if not prompt:
        return

    height, width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    max_text_width = int(width * max_width_frac) - 16
    words = prompt.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        (tw, _), _ = cv2.getTextSize(candidate, font, font_scale, thickness)
        if tw <= max_text_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if not lines:
        return
    # Cap to a few lines so the scene stays visible.
    lines = lines[:3]
    line_height = 18
    pad_y = 8
    banner_h = pad_y * 2 + line_height * len(lines)
    y0 = max(0, height - banner_h)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, y0), (width - 1, height - 1), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0.0, dst=image)
    for li, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (8, y0 + pad_y + line_height * (li + 1) - 4),
            font,
            font_scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )


# Per-episode trace behind the goal-pose figure: where the arm actually went, and where the
# decoded pose tokens said it should go. Written whenever POSE_TRAJ_DIR is set, alongside the
# camera matrix and a clean first frame, so the figure can be drawn offline without re-running
# the policy. Keyed by env index; one file per env, rewritten in place as the episode advances.
_POSE_TRAJ: dict[int, dict] = {}


def _dump_pose_traj(env_index: int, out_dir: str) -> None:
    import numpy as np

    rec = _POSE_TRAJ.get(env_index)
    if not rec or not rec["cur"]:
        return
    d = pathlib.Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f"traj_env{env_index}.npz",
        cur=np.asarray(rec["cur"], dtype=np.float64),
        pred=np.asarray(rec["pred"], dtype=np.float64),
        transform=np.asarray(rec["transform"], dtype=np.float64),
        first_frame=rec["first_frame"],
        init_state=rec.get("init_state", np.zeros(0)),
        states=np.asarray(rec.get("states", []), dtype=np.float64),
        bddl=np.array(rec.get("bddl", "")),
    )


def _maybe_overlay_goal_pose_frames(
    policy: PreTrainedPolicy,
    env: gym.vector.VectorEnv,
    frames: np.ndarray,
    *,
    preprocessor: PolicyProcessorPipeline | None = None,
) -> np.ndarray:
    """Overlay decoded MolmoAct2 goal poses onto rendered frames when possible.

    Requires SyncVectorEnv (direct sim access) and a policy that caches
    ``_last_goal_pose_norm`` via ``get_last_goal_pose_norm()``. Controlled by
    the ``POSE_VIZ_DIR`` environment variable in ``eval_policy``.
    """
    import cv2

    if not isinstance(env, gym.vector.SyncVectorEnv):
        raise RuntimeError("POSE_VIZ overlay currently requires SyncVectorEnv")

    batch_size, frame_height, frame_width = frames.shape[:3]
    panel_width = max(340, int(round(frame_height * 1.05)))
    out = np.full(
        (batch_size, frame_height, frame_width + panel_width, 3),
        18,
        dtype=frames.dtype,
    )
    out[:, :, :frame_width] = frames
    clean_frames = frames.copy()
    for batch_idx in range(batch_size):
        panel = out[batch_idx, :, frame_width:]
        cv2.putText(
            panel,
            "3D workspace (fixed view)",
            (14, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    pose = policy.get_last_goal_pose_norm() if hasattr(policy, "get_last_goal_pose_norm") else None
    from robosuite.utils.camera_utils import (
        get_camera_transform_matrix,
        project_points_from_world_to_camera,
    )

    pose_world = None
    if pose is not None:
        pose_np = pose.detach().float().cpu().numpy()
        if not np.isfinite(pose_np).all():
            raise RuntimeError("Decoded goal pose contains non-finite values.")
        pose_world = pose_np.astype(np.float64, copy=True)
    try:
        q01, q99 = _goal_pose_quantiles(policy, preprocessor)
    except RuntimeError:
        q01 = np.array([-0.5, -0.5, 0.0, -1.0, -1.0, -1.0], dtype=np.float64)
        q99 = np.array([0.5, 0.5, 0.55, 1.0, 1.0, 1.0], dtype=np.float64)
    if pose_world is not None:
        pose_world[..., :6] = (
            (pose_world[..., :6] + 1.0) * (q99[:6] - q01[:6]) / 2.0 + q01[:6]
        )

    def project(
        point: np.ndarray,
        transform: np.ndarray,
        height: int,
        width: int,
    ) -> tuple[int, int] | None:
        pixels = project_points_from_world_to_camera(
            np.asarray(point, dtype=np.float64).reshape(1, 3),
            transform,
            height,
            width,
        )[0]
        row, col = int(pixels[0]), int(pixels[1])
        row = height - 1 - row
        col = width - 1 - col
        if not (0 <= row < height and 0 <= col < width):
            return None
        return col, row

    def rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
        vector = np.asarray(axis_angle, dtype=np.float64).reshape(3)
        angle = float(np.linalg.norm(vector))
        if angle < 1e-8:
            return np.eye(3, dtype=np.float64)
        x, y, z = vector / angle
        cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)

    def draw_outlined_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        *,
        font_scale: float = 0.58,
        thickness: int = 2,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, origin, font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(image, text, origin, font, font_scale, color, thickness, cv2.LINE_AA)

    # Frames are RGB (write_video / gym render). OpenCV draws into that buffer
    # without converting, so these tuples must be RGB, not BGR.
    cur_rgb = (0, 220, 255)
    pred_rgb = (255, 36, 36)
    link_rgb = (255, 220, 0)
    axis_rgb = ((255, 48, 48), (36, 220, 64), (48, 96, 255))

    def draw_marker(
        image: np.ndarray,
        origin: tuple[int, int],
        color: tuple[int, int, int],
        marker: str,
    ) -> None:
        if marker == "square":
            half = 10
            cv2.rectangle(
                image,
                (origin[0] - half - 3, origin[1] - half - 3),
                (origin[0] + half + 3, origin[1] + half + 3),
                (255, 255, 255),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.rectangle(
                image,
                (origin[0] - half, origin[1] - half),
                (origin[0] + half, origin[1] + half),
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            return
        cv2.circle(image, origin, 14, (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
        cv2.circle(image, origin, 10, color, thickness=-1, lineType=cv2.LINE_AA)

    def draw_pose(
        image: np.ndarray,
        xyz: np.ndarray,
        rotation: np.ndarray,
        transform: np.ndarray,
        *,
        color: tuple[int, int, int],
        label: str,
        marker: str = "circle",
        axis_length: float = 0.08,
        draw_axes: bool = True,
        label_offset: tuple[int, int] = (14, -12),
    ) -> tuple[int, int] | None:
        height, width = image.shape[:2]
        origin = project(xyz, transform, height, width)
        if origin is None:
            return None
        if draw_axes:
            for axis, axis_color in enumerate(axis_rgb):
                tip = project(
                    np.asarray(xyz, dtype=np.float64) + axis_length * rotation[:, axis],
                    transform,
                    height,
                    width,
                )
                if tip is not None:
                    cv2.line(image, origin, tip, (0, 0, 0), 7, cv2.LINE_AA)
                    cv2.line(image, origin, tip, color, 5, cv2.LINE_AA)
                    cv2.line(image, origin, tip, axis_color, 3, cv2.LINE_AA)
        draw_marker(image, origin, color, marker)
        draw_outlined_text(
            image,
            label,
            (origin[0] + label_offset[0], max(22, origin[1] + label_offset[1])),
            color,
        )
        return origin

    def draw_dashed_line(
        image: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        *,
        width: int = 1,
        segments: int = 12,
    ) -> None:
        start_array = np.asarray(start, dtype=np.float64)
        end_array = np.asarray(end, dtype=np.float64)
        for segment in range(0, segments, 2):
            alpha0 = segment / segments
            alpha1 = min((segment + 1) / segments, 1.0)
            point0 = tuple(np.rint(start_array * (1.0 - alpha0) + end_array * alpha0).astype(int))
            point1 = tuple(np.rint(start_array * (1.0 - alpha1) + end_array * alpha1).astype(int))
            cv2.line(image, point0, point1, color, width, cv2.LINE_AA)

    def draw_3d_workspace(
        panel: np.ndarray,
        current_xyz: np.ndarray,
        current_rotation: np.ndarray,
        predicted_xyz: np.ndarray | None,
        predicted_rotation: np.ndarray | None,
    ) -> None:
        height, width = panel.shape[:2]
        lower = q01[:3].astype(np.float64)
        upper = q99[:3].astype(np.float64)
        margin = np.maximum((upper - lower) * 0.04, 1e-3)
        lower -= margin
        upper += margin
        center = (lower + upper) / 2.0

        azimuth = np.deg2rad(-55.0)
        elevation = np.deg2rad(25.0)
        view = np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
        up = np.cross(view, right)

        corners = np.array(
            [
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ],
            dtype=np.float64,
        )
        projected_corners = np.stack(
            ((corners - center) @ right, (corners - center) @ up),
            axis=-1,
        )
        projection_min = projected_corners.min(axis=0)
        projection_max = projected_corners.max(axis=0)
        projection_span = np.maximum(projection_max - projection_min, 1e-6)
        draw_left, draw_right = 18, width - 18
        draw_top, draw_bottom = 42, height - 72

        def project_3d(point: np.ndarray) -> tuple[int, int]:
            relative = np.asarray(point, dtype=np.float64) - center
            projected = np.array([relative @ right, relative @ up])
            normalized = np.clip(
                (projected - projection_min) / projection_span,
                0.0,
                1.0,
            )
            x = int(round(draw_left + normalized[0] * (draw_right - draw_left)))
            y = int(round(draw_bottom - normalized[1] * (draw_bottom - draw_top)))
            return x, y

        floor_color = (58, 58, 58)
        box_color = (88, 88, 88)
        for fraction in np.linspace(0.0, 1.0, 6):
            x = lower[0] + fraction * (upper[0] - lower[0])
            y = lower[1] + fraction * (upper[1] - lower[1])
            cv2.line(
                panel,
                project_3d(np.array([x, lower[1], lower[2]])),
                project_3d(np.array([x, upper[1], lower[2]])),
                floor_color,
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                panel,
                project_3d(np.array([lower[0], y, lower[2]])),
                project_3d(np.array([upper[0], y, lower[2]])),
                floor_color,
                1,
                cv2.LINE_AA,
            )

        corner_lookup = {
            (x_index, y_index, z_index): np.array([x, y, z])
            for x_index, x in enumerate((lower[0], upper[0]))
            for y_index, y in enumerate((lower[1], upper[1]))
            for z_index, z in enumerate((lower[2], upper[2]))
        }
        for x_index in range(2):
            for y_index in range(2):
                cv2.line(
                    panel,
                    project_3d(corner_lookup[(x_index, y_index, 0)]),
                    project_3d(corner_lookup[(x_index, y_index, 1)]),
                    box_color,
                    1,
                    cv2.LINE_AA,
                )
        for z_index in range(2):
            for fixed_axis in range(2):
                for fixed_value in range(2):
                    start_index = [0, 0, z_index]
                    end_index = [0, 0, z_index]
                    start_index[fixed_axis] = fixed_value
                    end_index[fixed_axis] = fixed_value
                    varying_axis = 1 - fixed_axis
                    start_index[varying_axis] = 0
                    end_index[varying_axis] = 1
                    cv2.line(
                        panel,
                        project_3d(corner_lookup[tuple(start_index)]),
                        project_3d(corner_lookup[tuple(end_index)]),
                        box_color,
                        1,
                        cv2.LINE_AA,
                    )

        axis_origin = lower.copy()
        axis_lengths = np.maximum((upper - lower) * 0.18, 0.035)
        for axis, (axis_name, axis_color) in enumerate(
            (("X", axis_rgb[0]), ("Y", axis_rgb[1]), ("Z", axis_rgb[2]))
        ):
            endpoint = axis_origin.copy()
            endpoint[axis] += axis_lengths[axis]
            start_pixel = project_3d(axis_origin)
            endpoint_pixel = project_3d(endpoint)
            cv2.arrowedLine(
                panel,
                start_pixel,
                endpoint_pixel,
                axis_color,
                2,
                cv2.LINE_AA,
                tipLength=0.18,
            )
            cv2.putText(
                panel,
                axis_name,
                (endpoint_pixel[0] + 3, endpoint_pixel[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                axis_color,
                1,
                cv2.LINE_AA,
            )

        markers = [(current_xyz, cur_rgb)]
        if predicted_xyz is not None:
            markers.append((predicted_xyz, pred_rgb))
        for xyz, color in markers:
            floor_point = np.array([xyz[0], xyz[1], lower[2]])
            draw_dashed_line(
                panel,
                project_3d(floor_point),
                project_3d(xyz),
                color,
                width=2,
            )

        current_pixel = project_3d(current_xyz)
        predicted_pixel = (
            project_3d(predicted_xyz) if predicted_xyz is not None else None
        )
        close_on_panel = (
            predicted_pixel is not None
            and float(np.linalg.norm(np.asarray(current_pixel) - np.asarray(predicted_pixel))) < 48.0
        )
        if predicted_pixel is not None:
            cv2.arrowedLine(
                panel,
                current_pixel,
                predicted_pixel,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
                tipLength=0.12,
            )
            cv2.arrowedLine(
                panel,
                current_pixel,
                predicted_pixel,
                link_rgb,
                3,
                cv2.LINE_AA,
                tipLength=0.12,
            )

        def draw_3d_pose(
            xyz: np.ndarray,
            rotation: np.ndarray,
            *,
            color: tuple[int, int, int],
            label: str,
            marker: str,
            axis_length: float,
            label_offset: tuple[int, int],
            draw_axes: bool = True,
        ) -> None:
            origin_pixel = project_3d(xyz)
            if draw_axes:
                for axis, axis_color in enumerate(axis_rgb):
                    endpoint = np.asarray(xyz) + axis_length * rotation[:, axis]
                    tip = project_3d(endpoint)
                    cv2.line(panel, origin_pixel, tip, (0, 0, 0), 5, cv2.LINE_AA)
                    cv2.line(panel, origin_pixel, tip, color, 4, cv2.LINE_AA)
                    cv2.arrowedLine(
                        panel,
                        origin_pixel,
                        tip,
                        axis_color,
                        2,
                        cv2.LINE_AA,
                        tipLength=0.16,
                    )
            draw_marker(panel, origin_pixel, color, marker)
            draw_outlined_text(
                panel,
                label,
                (
                    origin_pixel[0] + label_offset[0],
                    max(48, origin_pixel[1] + label_offset[1]),
                ),
                color,
                font_scale=0.5,
            )

        draw_3d_pose(
            current_xyz,
            current_rotation,
            color=cur_rgb,
            label="CUR",
            marker="circle",
            axis_length=0.05,
            label_offset=(-54, -18),
            draw_axes=not close_on_panel,
        )
        if predicted_xyz is not None and predicted_rotation is not None:
            draw_3d_pose(
                predicted_xyz,
                predicted_rotation,
                color=pred_rgb,
                label="PRED",
                marker="square",
                axis_length=0.08,
                label_offset=(16, 22) if close_on_panel else (16, -14),
            )
            position_error_cm = float(np.linalg.norm(predicted_xyz - current_xyz) * 100.0)
            draw_outlined_text(
                panel,
                f"CUR -> PRED: {position_error_cm:.1f} cm",
                (14, height - 42),
                (240, 240, 240),
                font_scale=0.48,
                thickness=1,
            )
            legend_3d = "cyan circle=current EE   red square=predicted t+10"
        else:
            legend_3d = "cyan circle=current EE   (no predicted goal pose)"
        draw_outlined_text(
            panel,
            legend_3d,
            (14, height - 18),
            (210, 210, 210),
            font_scale=0.36,
            thickness=1,
        )

    n = min(batch_size, len(env.envs))
    if pose_world is not None:
        n = min(n, int(pose_world.shape[0]))
    model_label = os.environ.get("POSE_VIZ_LABEL", "").strip()
    for i in range(n):
        sub = env.envs[i]
        inner = getattr(sub, "_env", None)
        if inner is None:
            continue
        sim = getattr(getattr(inner, "env", None), "sim", None) or getattr(inner, "sim", None)
        if sim is None:
            continue
        camera_frame = out[i, :, :frame_width]
        transform = get_camera_transform_matrix(
            sim,
            "agentview",
            frame_height,
            frame_width,
        )
        raw_obs = inner.env._get_observations()
        current_xyz = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float64)
        current_rotation = np.asarray(inner.robots[0].controller.ee_ori_mat, dtype=np.float64)
        predicted_xyz = None
        predicted_rotation = None
        if pose_world is not None:
            predicted_xyz = pose_world[i, :3]
            predicted_rotation = rotation_matrix(pose_world[i, 3:6])
        traj_dir = os.environ.get("POSE_TRAJ_DIR", "").strip()
        if traj_dir and predicted_xyz is not None:
            slot = _POSE_TRAJ.setdefault(
                i, {"cur": [], "pred": [], "transform": transform, "first_frame": clean_frames[i],
                    # the full MuJoCo state at t=0: lets the scene be rebuilt pixel-identical
                    # at a higher resolution than the policy ran at
                    "init_state": np.asarray(sim.get_state().flatten(), dtype=np.float64),
                    "states": [],
                    "bddl": str(getattr(inner, "bddl_file_name", "") or "")}
            )
            slot["cur"].append(np.asarray(current_xyz, dtype=np.float64).copy())
            slot["states"].append(np.asarray(sim.get_state().flatten(), dtype=np.float64))
            slot["pred"].append(np.asarray(predicted_xyz, dtype=np.float64).copy())
            slot["transform"] = transform
            _dump_pose_traj(i, traj_dir)
        cur_px = project(current_xyz, transform, frame_height, frame_width)
        pred_px = (
            project(predicted_xyz, transform, frame_height, frame_width)
            if predicted_xyz is not None
            else None
        )
        if cur_px is not None and pred_px is not None:
            cv2.arrowedLine(
                camera_frame,
                cur_px,
                pred_px,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
                tipLength=0.18,
            )
            cv2.arrowedLine(
                camera_frame,
                cur_px,
                pred_px,
                link_rgb,
                3,
                cv2.LINE_AA,
                tipLength=0.18,
            )
        close_on_image = (
            cur_px is not None
            and pred_px is not None
            and float(np.linalg.norm(np.asarray(cur_px) - np.asarray(pred_px))) < 70.0
        )
        draw_pose(
            camera_frame,
            current_xyz,
            current_rotation,
            transform,
            color=cur_rgb,
            label="CUR",
            marker="circle",
            axis_length=0.05,
            draw_axes=not close_on_image,
            label_offset=(-48, -16) if close_on_image else (14, -14),
        )
        if predicted_xyz is not None and predicted_rotation is not None:
            draw_pose(
                camera_frame,
                predicted_xyz,
                predicted_rotation,
                transform,
                color=pred_rgb,
                label="PRED",
                marker="square",
                axis_length=0.09,
                draw_axes=True,
                label_offset=(16, 24) if close_on_image else (16, -14),
            )
            legend = "CUR=cyan circle   PRED=red square (t+10)"
        else:
            legend = "CUR=cyan circle   (no predicted goal)"
        if model_label:
            legend = f"{model_label} | {legend}"
        cv2.rectangle(
            camera_frame,
            (4, 4),
            (min(frame_width - 4, 500), 28),
            (0, 0, 0),
            thickness=-1,
        )
        draw_outlined_text(camera_frame, legend, (8, 22), (255, 255, 255), font_scale=0.42, thickness=1)
        _draw_prompt_on_frame(camera_frame, _pose_viz_task_prompt(sub))
        draw_3d_workspace(
            out[i, :, frame_width:],
            current_xyz,
            current_rotation,
            predicted_xyz,
            predicted_rotation,
        )
    return out


def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv], None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_episode_count: int | None = None,
) -> dict:
    """Run a batched policy rollout once through a batch of environments.

    Note that all environments in the batch are run until the last environment is done. This means some
    data will probably need to be discarded (for environments that aren't the first one to be done).

    The return dictionary contains:
        (optional) "observation": A dictionary of (batch, sequence + 1, *) tensors mapped to observation
            keys. NOTE that this has an extra sequence element relative to the other keys in the
            dictionary. This is because an extra observation is included for after the environment is
            terminated or truncated.
        "action": A (batch, sequence, action_dim) tensor of actions applied based on the observations (not
            including the last observations).
        "reward": A (batch, sequence) tensor of rewards received for applying the actions.
        "success": A (batch, sequence) tensor of success conditions (the only time this can be True is upon
            environment termination/truncation).
        "done": A (batch, sequence) tensor of **cumulative** done conditions. For any given batch element,
            the first True is followed by True's all the way till the end. This can be used for masking
            extraneous elements from the sequences above.

    Args:
        env: The batch of environments.
        policy: The policy. Must be a PyTorch nn module.
        seeds: The environments are seeded once at the start of the rollout. If provided, this argument
            specifies the seeds for each of the environments.
        return_observations: Whether to include all observations in the returned rollout data. Observations
            are returned optionally because they typically take more memory to cache. Defaults to False.
        render_callback: Optional rendering callback to be used after the environments are reset, and after
            every step.
    Returns:
        The dictionary described above.
    """
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Reset the policy and environments.
    policy.reset()
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    while not np.all(done) and step < max_steps:
        # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
        observation = preprocess_observation(observation)
        if return_observations:
            all_observations.append(deepcopy(observation))

        # Infer "task" from sub-environments (prefer natural language description).
        # env.call() works with both SyncVectorEnv and AsyncVectorEnv.
        try:
            observation["task"] = list(env.call("task_description"))
        except (AttributeError, NotImplementedError):
            try:
                observation["task"] = list(env.call("task"))
            except (AttributeError, NotImplementedError):
                observation["task"] = [""] * env.num_envs

        # Apply environment-specific preprocessing (e.g., LiberoProcessorStep for LIBERO)
        observation = env_preprocessor(observation)

        observation = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = postprocessor(action)

        action_transition = {ACTION: action}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]

        # Convert to CPU / numpy.
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

        # Apply the next action.
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render_callback is not None:
            render_callback(env)

        # VectorEnv stores is_success in `info["final_info"][env_index]["is_success"]`. "final_info" isn't
        # available if none of the envs finished.
        if "final_info" in info:
            final_info = info["final_info"]
            if not isinstance(final_info, dict):
                raise RuntimeError(
                    "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                    "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                )
            successes = final_info["is_success"].tolist()
        elif "is_success" in info:
            is_success = info["is_success"]
            successes = (
                is_success.tolist() if hasattr(is_success, "tolist") else [bool(is_success)] * env.num_envs
            )
        else:
            successes = [False] * env.num_envs

        # Keep track of which environments are done so far.
        # Mark the episode as done if we reach the maximum step limit.
        # This ensures that the rollout always terminates cleanly at `max_steps`,
        # and allows logging/saving (e.g., videos) to be triggered consistently.
        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)

        all_actions.append(torch.from_numpy(action_numpy))
        all_rewards.append(torch.from_numpy(reward))
        all_dones.append(torch.from_numpy(done))
        all_successes.append(torch.tensor(successes))

        step += 1
        running_successes = einops.reduce(
            torch.stack(all_successes, dim=1), "b n -> b", "any"
        ).numpy()
        running_success_rate = running_successes.mean()
        progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
        progbar.update()
        if progress_callback is not None:
            episode_count = min(
                int(progress_episode_count or env.num_envs),
                int(env.num_envs),
            )
            success_count = int(running_successes[:episode_count].sum())
            progress_callback(
                {
                    "step": step,
                    "max_steps": int(max_steps),
                    "finished_rollouts": int(done[:episode_count].sum()),
                    "successes_so_far": success_count,
                    "total_rollouts": episode_count,
                    "running_success_rate": 100.0 * success_count / episode_count,
                }
            )

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    # Stack the sequence along the first dimension so that we have (batch, sequence, *) tensors.
    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        try:
            from peft import PeftModel

            if not isinstance(policy, PeftModel):
                raise exc
        except ImportError:
            raise exc from None

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        if isinstance(env, gym.vector.SyncVectorEnv):
            frames_now = np.stack([env.envs[i].render() for i in range(n_to_render_now)])  # noqa: B023
        elif hasattr(env, "call"):
            # Here we must render all frames and discard any we don't need.
            # Covers AsyncVectorEnv and _LazyAsyncVectorEnv (which wraps one).
            frames_now = np.stack(env.call("render")[:n_to_render_now])
        else:
            return

        # Optional pose overlay when POSE_VIZ_DIR is set.
        # Baseline has no predicted goal; the overlay still draws current EE.
        pose_viz_dir = os.environ.get("POSE_VIZ_DIR", "").strip()
        if pose_viz_dir:
            try:
                frames_now = _maybe_overlay_goal_pose_frames(
                    policy,
                    env,
                    frames_now,
                    preprocessor=preprocessor,
                )
            except Exception as exc:  # noqa: BLE001
                if not getattr(render_frame, "_pose_viz_warned", False):
                    logging.warning("POSE_VIZ overlay skipped: %s", exc)
                    render_frame._pose_viz_warned = True  # type: ignore[attr-defined]
        ep_frames.append(frames_now)

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        completed_before_batch = min(batch_ix * env.num_envs, n_episodes)
        episodes_in_batch = min(env.num_envs, n_episodes - completed_before_batch)
        successes_before_batch = sum(bool(value) for value in all_successes[:completed_before_batch])

        def report_batch_progress(payload: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            successes_so_far = successes_before_batch + int(payload["successes_so_far"])
            finished_rollouts = completed_before_batch + int(payload["finished_rollouts"])
            progress_callback(
                {
                    **payload,
                    "batch_index": batch_ix,
                    "successes_so_far": successes_so_far,
                    "finished_rollouts": min(finished_rollouts, n_episodes),
                    "total_rollouts": n_episodes,
                    # This is a monotonic lower bound while unfinished episodes can still succeed.
                    "running_success_rate": 100.0 * successes_so_far / n_episodes,
                }
            )

        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            progress_callback=report_batch_progress if progress_callback is not None else None,
            progress_episode_count=episodes_in_batch,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )

    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info(f"Making environment (batch_size={cfg.eval.batch_size}, async={cfg.eval.use_async_envs}).")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
        rename_map=cfg.rename_map,
    )

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        info = eval_policy_all(
            envs=envs,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            n_episodes=cfg.eval.n_episodes,
            max_episodes_rendered=cfg.eval.max_episodes_rendered,
            videos_dir=Path(cfg.output_dir) / "videos",
            start_seed=cfg.seed,
            max_parallel_tasks=cfg.env.max_parallel_tasks,
            live_output_path=Path(cfg.output_dir) / "live_eval.json",
        )
        print("Overall Aggregated Metrics:")
        print(info["overall"])

        # Print per-suite stats
        for task_group, task_group_info in info.items():
            print(f"\nAggregated Metrics for {task_group}:")
            print(task_group_info)
    # Close all vec envs
    close_envs(envs)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        progress_callback=progress_callback,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    progress_callback: Callable[[str, int, dict[str, Any]], None] | None = None,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    task_progress_callback = None
    if progress_callback is not None:
        task_progress_callback = lambda payload: progress_callback(task_group, task_id, payload)

    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        progress_callback=task_progress_callback,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
    live_output_path: Path | None = None,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []
    running_tasks: dict[str, dict[str, Any]] = {}
    live_lock = threading.Lock()
    last_live_write = 0.0

    def _aggregate(acc: dict[str, list]) -> dict:
        successes = acc["successes"]
        elapsed = time.time() - start_t
        return {
            "avg_sum_reward": float(np.mean(acc["sum_rewards"])) if acc["sum_rewards"] else float("nan"),
            "avg_max_reward": float(np.mean(acc["max_rewards"])) if acc["max_rewards"] else float("nan"),
            "pc_success": float(np.mean(successes) * 100) if successes else float("nan"),
            "n_success": int(np.sum(successes)) if successes else 0,
            "n_episodes": len(acc["sum_rewards"]),
            "eval_s": elapsed,
        }

    def _write_live_unlocked(status: str) -> None:
        if live_output_path is None:
            return
        payload = {
            "status": status,
            "completed_tasks": len(per_task_infos),
            "total_tasks": len(tasks),
            "running_tasks": list(running_tasks.values()),
            "per_task": per_task_infos,
            "per_group": {group: _aggregate(acc) for group, acc in group_acc.items()},
            "overall": _aggregate(overall),
            "updated_at": time.time(),
        }
        live_output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = live_output_path.with_suffix(f"{live_output_path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(live_output_path)

        realtime_path = live_output_path.parent / "realtime_accuracy.json"
        realtime_payload = {
            "status": status,
            "completed_tasks": len(per_task_infos),
            "total_tasks": len(tasks),
            "completed_rollouts": _aggregate(overall),
            "running_tasks": list(running_tasks.values()),
            "note": (
                "running_success_rate is a lower bound over all configured rollouts; "
                "unfinished rollouts may still succeed."
            ),
            "updated_at": time.time(),
        }
        realtime_tmp = realtime_path.with_suffix(f"{realtime_path.suffix}.tmp")
        realtime_tmp.write_text(json.dumps(realtime_payload, indent=2) + "\n", encoding="utf-8")
        realtime_tmp.replace(realtime_path)

    def _write_live(status: str) -> None:
        with live_lock:
            _write_live_unlocked(status)

    def _update_running_task(task_group: str, task_id: int, progress: dict[str, Any]) -> None:
        nonlocal last_live_write
        now = time.time()
        key = f"{task_group}:{task_id}"
        with live_lock:
            previous = running_tasks.get(key, {})
            running_tasks[key] = {
                "task_group": task_group,
                "task_id": task_id,
                **progress,
                "accuracy_is_lower_bound": True,
                "updated_at": now,
            }
            changed = (
                previous.get("successes_so_far") != progress.get("successes_so_far")
                or previous.get("finished_rollouts") != progress.get("finished_rollouts")
            )
            if changed or now - last_live_write >= 2.0:
                _write_live_unlocked("running")
                last_live_write = now

    def _finish_running_task(task_group: str, task_id: int) -> None:
        with live_lock:
            running_tasks.pop(f"{task_group}:{task_id}", None)

    _write_live("running")

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        progress_callback=_update_running_task,
    )

    if max_parallel_tasks <= 1:
        prefetch_thread: threading.Thread | None = None
        for i, (task_group, task_id, env) in enumerate(tasks):
            if prefetch_thread is not None:
                prefetch_thread.join()
                prefetch_thread = None

            try:
                tg, tid, metrics = task_runner(task_group, task_id, env)
                _finish_running_task(tg, tid)
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
                _write_live("running")
            finally:
                env.close()
                # Prefetch next task's workers *after* closing current env to prevent
                # GPU memory overlap between consecutive tasks.
                if i + 1 < len(tasks):
                    next_env = tasks[i + 1][2]
                    if hasattr(next_env, "_ensure"):
                        prefetch_thread = threading.Thread(target=next_env._ensure, daemon=True)
                        prefetch_thread.start()
    else:
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id, env)
            for fut in cf.as_completed(fut2meta):
                tg, tid, env = fut2meta[fut]
                try:
                    tg, tid, metrics = fut.result()
                    _finish_running_task(tg, tid)
                    _accumulate_to(tg, metrics)
                    per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
                    _write_live("running")
                finally:
                    env.close()

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    _write_live("final")

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }


def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
