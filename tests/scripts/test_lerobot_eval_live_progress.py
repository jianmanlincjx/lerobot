import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.scripts import lerobot_eval


class _DummyEnv:
    def close(self):
        pass


def test_goal_pose_quantiles_come_from_loaded_preprocessor():
    policy = SimpleNamespace(
        config=SimpleNamespace(goal_pose_feature_key="observation.state", dataset_stats=None)
    )
    preprocessor = SimpleNamespace(
        steps=[
            SimpleNamespace(stats=None),
            SimpleNamespace(
                stats={
                    "observation.state": {
                        "q01": torch.tensor([-0.4, -0.2, 0.04, 1.5, -2.7, -1.1]),
                        "q99": torch.tensor([0.1, 0.3, 1.27, 3.3, 2.4, 0.6]),
                    }
                }
            ),
        ]
    )

    q01, q99 = lerobot_eval._goal_pose_quantiles(policy, preprocessor)

    np.testing.assert_allclose(q01, [-0.4, -0.2, 0.04, 1.5, -2.7, -1.1])
    np.testing.assert_allclose(q99, [0.1, 0.3, 1.27, 3.3, 2.4, 0.6])


def test_goal_pose_quantiles_fail_closed_without_stats():
    policy = SimpleNamespace(
        config=SimpleNamespace(goal_pose_feature_key="observation.state", dataset_stats=None)
    )
    preprocessor = SimpleNamespace(steps=[SimpleNamespace(stats=None)])

    with pytest.raises(RuntimeError, match="without saved q01/q99"):
        lerobot_eval._goal_pose_quantiles(policy, preprocessor)


def test_eval_policy_all_writes_realtime_task_accuracy(monkeypatch, tmp_path):
    live_path = tmp_path / "live_eval.json"
    realtime_path = tmp_path / "realtime_accuracy.json"

    def fake_run_one(task_group, task_id, env, *, progress_callback, **kwargs):
        del env, kwargs
        progress_callback(
            task_group,
            task_id,
            {
                "step": 40,
                "max_steps": 280,
                "finished_rollouts": 2,
                "successes_so_far": 2,
                "total_rollouts": 4,
                "running_success_rate": 50.0,
            },
        )
        running = json.loads(realtime_path.read_text())
        assert running["running_tasks"] == [
            {
                "task_group": task_group,
                "task_id": task_id,
                "step": 40,
                "max_steps": 280,
                "finished_rollouts": 2,
                "successes_so_far": 2,
                "total_rollouts": 4,
                "running_success_rate": 50.0,
                "accuracy_is_lower_bound": True,
                "updated_at": running["running_tasks"][0]["updated_at"],
            }
        ]
        return task_group, task_id, {
            "sum_rewards": [1.0, 1.0, 0.0, 0.0],
            "max_rewards": [1.0, 1.0, 0.0, 0.0],
            "successes": [True, True, False, False],
            "video_paths": [],
        }

    monkeypatch.setattr(lerobot_eval, "run_one", fake_run_one)
    result = lerobot_eval.eval_policy_all(
        envs={"libero_spatial": {0: _DummyEnv()}},
        policy=None,
        env_preprocessor=None,
        env_postprocessor=None,
        preprocessor=None,
        postprocessor=None,
        n_episodes=4,
        live_output_path=live_path,
    )

    final = json.loads(realtime_path.read_text())
    assert final["status"] == "final"
    assert final["running_tasks"] == []
    assert final["completed_rollouts"]["n_episodes"] == 4
    assert final["completed_rollouts"]["pc_success"] == 50.0
    assert result["overall"]["pc_success"] == 50.0
