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
"""Contract tests for DatasetReader."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import get_safe_default_codec


def _write_sample_manifest(path, rows):
    pq.write_table(
        pa.table(
            {
                "index": [row["index"] for row in rows],
                "episode_index": [row["episode_index"] for row in rows],
                "frame_index": [row["frame_index"] for row in rows],
            }
        ),
        path,
    )

# ── Loading ──────────────────────────────────────────────────────────


def test_try_load_returns_true_when_data_exists(tmp_path, lerobot_dataset_factory):
    """Given a fully written dataset, try_load() returns True."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=20, use_videos=False
    )
    reader = DatasetReader(
        meta=dataset.meta,
        root=dataset.root,
        episodes=None,
        tolerance_s=1e-4,
        video_backend=get_safe_default_codec(),
        delta_timestamps=None,
        image_transforms=None,
    )
    assert reader.try_load() is True
    assert reader.hf_dataset is not None


def test_try_load_returns_false_when_no_data(tmp_path):
    """When only metadata exists (no data/ parquets), try_load() returns False."""
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    root = tmp_path / "meta_only"
    features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
    meta = LeRobotDatasetMetadata.create(
        repo_id="test/meta_only", fps=30, features=features, root=root, use_videos=False
    )

    reader = DatasetReader(
        meta=meta,
        root=meta.root,
        episodes=None,
        tolerance_s=1e-4,
        video_backend=get_safe_default_codec(),
        delta_timestamps=None,
        image_transforms=None,
    )
    assert reader.try_load() is False
    assert reader.hf_dataset is None


# ── Counts ───────────────────────────────────────────────────────────


def test_num_frames_without_filter(tmp_path, lerobot_dataset_factory):
    """With episodes=None, num_frames equals total_frames."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=3, total_frames=60, use_videos=False
    )
    assert dataset.reader.num_frames == dataset.meta.total_frames


def test_num_episodes_without_filter(tmp_path, lerobot_dataset_factory):
    """With episodes=None, num_episodes equals total_episodes."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=3, total_frames=60, use_videos=False
    )
    assert dataset.reader.num_episodes == dataset.meta.total_episodes


def test_num_frames_with_episode_filter(tmp_path, lerobot_dataset_factory):
    """When filtering to a subset, only those episodes' frames are counted."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=5, total_frames=100, episodes=[0, 2], use_videos=False
    )
    # Filtered frames should be less than total
    assert dataset.reader.num_frames <= dataset.meta.total_frames
    assert dataset.reader.num_episodes == 2


# ── get_item ─────────────────────────────────────────────────────────


def test_get_item_returns_expected_keys(tmp_path, lerobot_dataset_factory):
    """get_item(0) returns a dict with expected keys."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=1, total_frames=10, use_videos=False
    )
    item = dataset.reader.get_item(0)

    # Standard keys that must always be present
    for key in ["index", "episode_index", "frame_index", "timestamp", "task_index", "task"]:
        assert key in item, f"Missing key: {key}"


def test_get_item_values_are_correct(tmp_path, lerobot_dataset_factory):
    """get_item() returns correct index and episode_index."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=20, use_videos=False
    )
    item_0 = dataset.reader.get_item(0)

    assert item_0["index"].item() == 0
    assert item_0["episode_index"].item() == 0


# ── Transforms ───────────────────────────────────────────────────────


def test_image_transforms_are_applied(tmp_path, lerobot_dataset_factory):
    """When image_transforms is provided, get_item() applies it to camera keys."""
    transform_called = {"count": 0}

    def sentinel_transform(img):
        transform_called["count"] += 1
        return img

    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds",
        total_episodes=1,
        total_frames=5,
        use_videos=False,
        image_transforms=sentinel_transform,
    )
    item = dataset[0]  # noqa: F841

    # Should have been called once per camera key per frame
    num_cameras = len(dataset.meta.camera_keys)
    if num_cameras > 0:
        assert transform_called["count"] >= 1


# ── File paths ───────────────────────────────────────────────────────


def test_get_episodes_file_paths_returns_data_paths(tmp_path, lerobot_dataset_factory):
    """get_episodes_file_paths() returns paths including data/ paths."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=20, use_videos=False
    )
    paths = dataset.reader.get_episodes_file_paths()

    assert len(paths) > 0
    assert any("data/" in str(p) for p in paths)


def test_get_episodes_file_paths_includes_video_paths(tmp_path, lerobot_dataset_factory):
    """When dataset has video keys, file paths include video/ paths."""
    dataset = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=20, use_videos=True
    )

    if len(dataset.meta.video_keys) > 0:
        paths = dataset.reader.get_episodes_file_paths()
        assert any("video" in str(p).lower() for p in paths)


def test_sample_manifest_maps_public_indices_and_preserves_absolute_delta_queries(
    tmp_path, lerobot_dataset_factory
):
    base = lerobot_dataset_factory(
        root=tmp_path / "ds", total_episodes=2, total_frames=20, use_videos=False
    )
    anchors = []
    for abs_idx in range(len(base)):
        row = base.get_raw_item(abs_idx)
        if int(row["frame_index"]) == 1:
            anchors.append(
                {
                    "index": int(row["index"]),
                    "episode_index": int(row["episode_index"]),
                    "frame_index": int(row["frame_index"]),
                }
            )
    anchors.reverse()
    manifest_path = tmp_path / "anchors.parquet"
    _write_sample_manifest(manifest_path, anchors)

    sampled = LeRobotDataset(
        base.repo_id,
        root=base.root,
        sample_indices_path=manifest_path,
        delta_timestamps={ACTION: [0.0, 1.0 / base.fps]},
    )

    assert len(sampled) == len(anchors) == 2
    assert [int(sampled[idx]["index"]) for idx in range(len(sampled))] == [
        row["index"] for row in anchors
    ]
    assert int(sampled.get_raw_item(0)["index"]) == anchors[0]["index"]
    assert sampled.select_columns("index")[:]["index"] == [row["index"] for row in anchors]
    first = sampled[0]
    anchor_idx = anchors[0]["index"]
    assert torch.equal(first[ACTION][0], base.get_raw_item(anchor_idx)[ACTION])
    assert torch.equal(first[ACTION][1], base.get_raw_item(anchor_idx + 1)[ACTION])

    episode_filtered = LeRobotDataset(
        base.repo_id,
        root=base.root,
        episodes=[1],
        sample_indices_path=manifest_path,
        delta_timestamps={ACTION: [0.0, 1.0 / base.fps]},
    )
    assert len(episode_filtered) == 1
    assert int(episode_filtered[0]["episode_index"]) == 1


@pytest.mark.parametrize("invalid_kind", ["duplicate", "out_of_range", "mismatch"])
def test_sample_manifest_strict_validation(tmp_path, lerobot_dataset_factory, invalid_kind):
    base = lerobot_dataset_factory(
        root=tmp_path / invalid_kind,
        total_episodes=2,
        total_frames=20,
        use_videos=False,
    )
    raw = base.get_raw_item(1)
    row = {
        "index": int(raw["index"]),
        "episode_index": int(raw["episode_index"]),
        "frame_index": int(raw["frame_index"]),
    }
    rows = [row]
    if invalid_kind == "duplicate":
        rows.append(row.copy())
    elif invalid_kind == "out_of_range":
        rows[0]["index"] = base.meta.total_frames
    else:
        rows[0]["frame_index"] += 1
    manifest_path = tmp_path / f"{invalid_kind}.parquet"
    _write_sample_manifest(manifest_path, rows)

    with pytest.raises(ValueError):
        LeRobotDataset(base.repo_id, root=base.root, sample_indices_path=manifest_path)
