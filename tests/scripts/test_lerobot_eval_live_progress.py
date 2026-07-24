import json

from lerobot.scripts import lerobot_eval


class _DummyEnv:
    def close(self):
        pass


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
