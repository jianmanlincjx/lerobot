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
"""Private reader component for LeRobotDataset. Handles random-access reading (HF dataset, delta indices, video decoding)."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
import pyarrow.parquet as pq
import torch

from .dataset_metadata import LeRobotDatasetMetadata
from .feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from .io_utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)
from .video_utils import decode_video_frames


@dataclass(frozen=True)
class SampleManifest:
    indices: np.ndarray
    episode_indices: np.ndarray
    frame_indices: np.ndarray

    def __len__(self) -> int:
        return len(self.indices)


class DatasetReader:
    """Encapsulates read-side state and methods for LeRobotDataset.

    Owns: hf_dataset, _absolute_to_relative_idx, delta_indices.
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        return_uint8: bool = False,
        sample_indices_path: str | Path | None = None,
        skip_video_decode: bool = False,
    ):
        """Initialize the reader with metadata, filtering, and transform config.

        The HF dataset is not loaded here — call :meth:`try_load` or
        :meth:`load_and_activate` afterward.

        Args:
            meta: Dataset metadata instance.
            root: Local dataset root directory.
            episodes: Optional list of episode indices to select. ``None``
                means all episodes.
            tolerance_s: Timestamp synchronization tolerance in seconds.
            video_backend: Video decoding backend identifier.
            delta_timestamps: Optional dict mapping feature keys to lists of
                relative timestamp offsets for temporal context windows.
            image_transforms: Optional torchvision v2 transform applied to
                visual features.
        """
        self._meta = meta
        self.root = root
        self.episodes = episodes
        self._tolerance_s = tolerance_s
        self._video_backend = video_backend
        self._image_transforms = image_transforms
        self._return_uint8 = return_uint8
        self._skip_video_decode = skip_video_decode
        self.sample_indices_path = Path(sample_indices_path).expanduser() if sample_indices_path else None

        self.hf_dataset: datasets.Dataset | None = None
        self._absolute_to_relative_idx: dict[int, int] | None = None
        self._sample_manifest_rows = self._load_sample_manifest()
        self._sample_relative_indices: np.ndarray | None = None

        # Setup delta_indices (doesn't depend on hf_dataset)
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)

    def try_load(self) -> bool:
        """Attempt to load from local cache. Returns True if data is sufficient."""
        try:
            self.hf_dataset = self._load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError):
            self.hf_dataset = None
            return False
        if not self._check_cached_episodes_sufficient():
            self.hf_dataset = None
            return False
        self._build_index_mapping()
        return True

    def load_and_activate(self) -> None:
        """Load HF dataset from disk and build index mapping. Call after data is on disk."""
        self.hf_dataset = self._load_hf_dataset()
        self._build_index_mapping()

    def _build_index_mapping(self) -> None:
        """Build absolute-to-relative index mapping from loaded hf_dataset."""
        self._absolute_to_relative_idx = None
        self._sample_relative_indices = None
        if (self.episodes is not None or self._sample_manifest_rows is not None) and self.hf_dataset is not None:
            indices = self.hf_dataset.data.column("index").to_numpy()
            identity_mapping = (
                self.episodes is None
                and len(indices) == self._meta.total_frames
                and (len(indices) == 0 or (int(indices[0]) == 0 and int(indices[-1]) == len(indices) - 1))
                and (len(indices) < 2 or np.all(np.diff(indices) == 1))
            )
            if not identity_mapping:
                self._absolute_to_relative_idx = dict(
                    zip(indices.tolist(), range(len(indices)), strict=True)
                )
        if self._sample_manifest_rows is not None:
            self._validate_and_map_sample_manifest()

    def _load_sample_manifest(self) -> SampleManifest | None:
        """Load and metadata-validate a read-only anchor manifest."""
        if self.sample_indices_path is None:
            return None
        if not self.sample_indices_path.is_file():
            raise FileNotFoundError(f"Sample indices manifest not found: {self.sample_indices_path}")

        required = {"index", "episode_index", "frame_index"}
        schema = pq.read_schema(self.sample_indices_path)
        missing = sorted(required.difference(schema.names))
        if missing:
            raise ValueError(
                f"Sample indices manifest {self.sample_indices_path} is missing columns: {missing}."
            )
        table = pq.read_table(self.sample_indices_path, columns=sorted(required))
        if table.num_rows == 0:
            raise ValueError(f"Sample indices manifest {self.sample_indices_path} is empty.")

        columns: dict[str, np.ndarray] = {}
        for key in required:
            column = table.column(key).combine_chunks()
            if column.null_count or not np.issubdtype(column.to_numpy(zero_copy_only=False).dtype, np.integer):
                raise ValueError(
                    f"Sample indices manifest column {key!r} must contain non-null integers."
                )
            columns[key] = column.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)

        indices = columns["index"]
        episode_indices = columns["episode_index"]
        frame_indices = columns["frame_index"]
        if len(indices) > 1:
            differences = np.diff(indices)
            monotonic_unique = np.all(differences > 0) or np.all(differences < 0)
            if not monotonic_unique:
                unique_indices, counts = np.unique(indices, return_counts=True)
                duplicates = unique_indices[counts > 1]
                if duplicates.size:
                    raise ValueError(
                        f"Sample indices manifest contains duplicate index {int(duplicates[0])}."
                    )

        invalid_index = np.flatnonzero((indices < 0) | (indices >= self._meta.total_frames))
        if invalid_index.size:
            value = int(indices[invalid_index[0]])
            raise ValueError(
                f"Sample indices manifest index {value} is outside [0, {self._meta.total_frames})."
            )
        invalid_episode = np.flatnonzero(
            (episode_indices < 0) | (episode_indices >= self._meta.total_episodes)
        )
        if invalid_episode.size:
            value = int(episode_indices[invalid_episode[0]])
            raise ValueError(
                f"Sample indices manifest episode_index {value} is outside "
                f"[0, {self._meta.total_episodes})."
            )

        episode_starts = np.fromiter(
            (
                int(self._meta.episodes[episode_index]["dataset_from_index"])
                for episode_index in range(self._meta.total_episodes)
            ),
            dtype=np.int64,
            count=self._meta.total_episodes,
        )
        episode_ends = np.fromiter(
            (
                int(self._meta.episodes[episode_index]["dataset_to_index"])
                for episode_index in range(self._meta.total_episodes)
            ),
            dtype=np.int64,
            count=self._meta.total_episodes,
        )
        starts = episode_starts[episode_indices]
        ends = episode_ends[episode_indices]
        expected_frames = indices - starts
        mismatch = np.flatnonzero(
            (indices < starts) | (indices >= ends) | (frame_indices != expected_frames)
        )
        if mismatch.size:
            row_idx = int(mismatch[0])
            raise ValueError(
                f"Sample indices manifest row {row_idx} does not match dataset metadata: "
                f"index={int(indices[row_idx])}, "
                f"episode_index={int(episode_indices[row_idx])}, "
                f"frame_index={int(frame_indices[row_idx])}; "
                f"expected index in [{int(starts[row_idx])}, {int(ends[row_idx])}) "
                f"and frame_index={int(expected_frames[row_idx])}."
            )

        if self.episodes is not None:
            selected = np.isin(episode_indices, np.asarray(self.episodes, dtype=np.int64))
            indices = indices[selected]
            episode_indices = episode_indices[selected]
            frame_indices = frame_indices[selected]
        if not len(indices):
            raise ValueError("Sample indices manifest has no rows after applying the episodes filter.")
        return SampleManifest(indices, episode_indices, frame_indices)

    def _validate_and_map_sample_manifest(self) -> None:
        """Validate manifest rows against loaded parquet values and build public-index mapping."""
        if self.hf_dataset is None or self._sample_manifest_rows is None:
            return
        manifest = self._sample_manifest_rows
        episode_values = self.hf_dataset.data.column("episode_index").to_numpy()
        frame_values = self.hf_dataset.data.column("frame_index").to_numpy()
        if self._absolute_to_relative_idx is None:
            relative_indices = manifest.indices
        else:
            relative_indices = np.fromiter(
                (
                    self._absolute_to_relative_idx.get(int(abs_idx), -1)
                    for abs_idx in manifest.indices
                ),
                dtype=np.int64,
                count=len(manifest),
            )
            missing = np.flatnonzero(relative_indices < 0)
            if missing.size:
                abs_idx = int(manifest.indices[missing[0]])
                raise ValueError(
                    f"Sample indices manifest index {abs_idx} is unavailable after applying episodes filter."
                )
        mismatch = np.flatnonzero(
            (episode_values[relative_indices] != manifest.episode_indices)
            | (frame_values[relative_indices] != manifest.frame_indices)
        )
        if mismatch.size:
            row_idx = int(mismatch[0])
            relative_idx = int(relative_indices[row_idx])
            actual = (int(episode_values[relative_idx]), int(frame_values[relative_idx]))
            expected = (
                int(manifest.episode_indices[row_idx]),
                int(manifest.frame_indices[row_idx]),
            )
            raise ValueError(
                f"Sample indices manifest index {int(manifest.indices[row_idx])} "
                f"maps to {actual}, expected {expected}."
            )
        self._sample_relative_indices = relative_indices

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        if self._sample_manifest_rows is not None:
            return len(self._sample_manifest_rows)
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self._meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        if self._sample_manifest_rows is not None:
            return int(np.unique(self._sample_manifest_rows.episode_indices).size)
        return len(self.episodes) if self.episodes is not None else self._meta.total_episodes

    def _load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        features = get_hf_features_from_features(self._meta.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes and their video files."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        if self.episodes is None:
            requested_episodes = set(range(self._meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        if not requested_episodes.issubset(available_episodes):
            return False

        if len(self._meta.video_keys) > 0:
            for ep_idx in requested_episodes:
                for vid_key in self._meta.video_keys:
                    video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def get_episodes_file_paths(self) -> list[Path]:
        """Return deduplicated file paths (data + video) for selected episodes.

        Used to build the ``allow_patterns`` list for ``snapshot_download``.
        """
        episodes = self.episodes if self.episodes is not None else list(range(self._meta.total_episodes))
        fpaths = [str(self._meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self._meta.video_keys) > 0:
            video_files = [
                str(self._meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self._meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episodes are stored in the same files, so we return unique paths only
        fpaths = list(set(fpaths))
        return fpaths

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Compute query indices for delta timestamps."""
        ep = self._meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self._meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Query dataset for indices across keys, skipping video keys."""
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self._meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault.
        """
        ep = self._meta.episodes[ep_idx]

        def _decode_single(vid_key: str, query_ts: list[float]) -> tuple[str, torch.Tensor]:
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(
                video_path,
                shifted_query_ts,
                self._tolerance_s,
                self._video_backend,
                return_uint8=self._return_uint8,
            )
            return vid_key, frames.squeeze(0)

        items = list(query_timestamps.items())

        # Single camera: no threading overhead
        if len(items) <= 1:
            return {vid_key: _decode_single(vid_key, query_ts)[1] for vid_key, query_ts in items}

        # Multi-camera: decode in parallel (video decoding releases the GIL)
        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            futures = [pool.submit(_decode_single, k, ts) for k, ts in items]
            return dict(f.result() for f in futures)

    def _resolve_public_index(self, idx: int) -> int:
        if self._sample_relative_indices is None:
            return idx
        idx = int(idx)
        if not -len(self._sample_relative_indices) <= idx < len(self._sample_relative_indices):
            raise IndexError(idx)
        return self._sample_relative_indices[idx]

    def select_columns(self, column_names: str | list[str]) -> datasets.Dataset:
        """Select columns while preserving the public manifest subset and order."""
        dataset = self.hf_dataset
        if self._sample_relative_indices is not None:
            dataset = dataset.select(self._sample_relative_indices)
        return dataset.select_columns(column_names)

    def get_raw_item(self, idx: int) -> dict:
        """Get a raw row through the public manifest index mapping."""
        return self.hf_dataset[self._resolve_public_index(idx)]

    def get_item(self, idx) -> dict:
        """Core __getitem__ logic. Assumes hf_dataset is loaded.

        ``idx`` is a *relative* index into the (possibly episode-filtered)
        HF dataset, **not** the absolute frame index stored in the ``index``
        column.  The absolute index is retrieved from the row itself.
        """
        item = self.hf_dataset[self._resolve_public_index(idx)]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self._meta.video_keys) > 0 and not self._skip_video_decode:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self._image_transforms is not None:
            image_keys = self._meta.camera_keys
            for cam in image_keys:
                item[cam] = self._image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name

        # add subtask information if available
        if "subtask_index" in self._meta.features and self._meta.subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self._meta.subtasks.iloc[subtask_idx].name

        return item
