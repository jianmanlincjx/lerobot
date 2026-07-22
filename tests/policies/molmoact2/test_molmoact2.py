from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import save_file

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies import get_policy_class, make_policy_config
from lerobot.policies.molmoact2 import modeling_molmoact2 as molmoact2_modeling
from lerobot.policies.molmoact2.configuration_molmoact2 import (
    MolmoAct2Config,
    MolmoAct2CosineDecayWithWarmupSchedulerConfig,
    infer_molmoact2_max_sequence_length,
)
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.policies.molmoact2.processor_molmoact2 import (
    MolmoAct2MaskedNormalizerProcessorStep,
    MolmoAct2MaskedUnnormalizerProcessorStep,
    MolmoAct2PackInputsProcessorStep,
    _add_gripper_masks_to_stats,
    _build_discrete_state_string,
    _normalize_question_text,
)
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def test_molmoact2_policy_registration():
    cfg = make_policy_config("molmoact2", checkpoint_path="/tmp/not-a-real-checkpoint")

    assert cfg.type == "molmoact2"
    assert cfg.action_mode == "both"
    assert cfg.normalize_gripper is False
    assert cfg.enable_knowledge_insulation is False
    assert cfg.freeze_embedding is True
    assert cfg.per_episode_seed is False
    assert cfg.eval_seed is None
    assert cfg.normalize_language is True
    assert cfg.get_scheduler_preset().num_decay_steps is None
    assert cfg.action_delta_indices == list(range(cfg.chunk_size))
    assert get_policy_class("molmoact2") is MolmoAct2Policy


def test_molmoact2_scheduler_decay_steps_auto_match_training_steps():
    param = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([param], lr=0.001)
    config = MolmoAct2CosineDecayWithWarmupSchedulerConfig(
        peak_lr=0.01,
        decay_lr=0.001,
        num_warmup_steps=10,
        num_decay_steps=None,
    )

    scheduler = config.build(optimizer, num_training_steps=100)
    for _ in range(100):
        optimizer.step()
        scheduler.step()

    assert scheduler.get_last_lr() == pytest.approx([0.0001])


def test_molmoact2_scheduler_uses_independent_group_warmups():
    params = [torch.nn.Parameter(torch.ones(())) for _ in range(2)]
    optimizer = torch.optim.AdamW(
        [
            {"params": [params[0]], "lr": 1e-5, "name": "vlm"},
            {"params": [params[1]], "lr": 5e-5, "name": "action_expert"},
        ]
    )
    config = MolmoAct2CosineDecayWithWarmupSchedulerConfig(
        peak_lr=1e-5,
        decay_lr=1e-6,
        num_warmup_steps=10,
        num_decay_steps=100,
        vlm_warmup_steps=10,
        action_expert_warmup_steps=2,
    )

    scheduler = config.build(optimizer, num_training_steps=100)

    assert scheduler.get_last_lr() == pytest.approx([1e-5 / 11, 5e-5 / 3])
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[1] == pytest.approx(5e-5)
    assert scheduler.get_last_lr()[0] < 1e-5


def test_clean_vlm_bootstrap_loads_vlm_and_resets_action_expert(tmp_path, monkeypatch):
    class DummyActionExpert(torch.nn.Linear):
        def reset_parameters(self):
            torch.nn.init.constant_(self.weight, 0.25)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = torch.nn.Linear(2, 2, bias=False)
            self.embedding = torch.nn.Parameter(torch.zeros(4, 2))
            self.action_expert = DummyActionExpert(2, 2)

        def _require_action_expert(self):
            return self.action_expert

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyBackbone()

    save_file(
        {
            "model.body.weight": torch.full((2, 2), 3.0),
            "model.embedding": torch.full((3, 2), 4.0),
        },
        tmp_path / "model.safetensors",
    )
    monkeypatch.setattr(
        molmoact2_modeling,
        "_resolve_checkpoint_location",
        lambda checkpoint_path, **kwargs: str(tmp_path),
    )
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.model = DummyModel()
    torch.nn.init.constant_(policy.model.model.action_expert.weight, 9.0)
    policy.config = SimpleNamespace(
        vlm_checkpoint_path=str(tmp_path),
        vlm_checkpoint_revision=None,
        vlm_checkpoint_force_download=False,
        audit_bootstrap=False,
    )

    policy._apply_clean_bootstrap()

    assert torch.equal(policy.model.model.body.weight, torch.full((2, 2), 3.0))
    assert torch.equal(policy.model.model.embedding[:3], torch.full((3, 2), 4.0))
    assert torch.equal(policy.model.model.embedding[3], torch.zeros(2))
    assert torch.equal(policy.model.model.action_expert.weight, torch.full((2, 2), 0.25))
    assert policy.bootstrap_audit["loaded_exact"] == 1
    assert policy.bootstrap_audit["loaded_partial"] == 1


def test_disable_visual_input_does_not_require_camera_keys():
    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.disable_visual_input = True
    step.image_keys = ["observation.images.image", "observation.images.image2"]

    assert step._resolve_image_keys({OBS_STATE: torch.zeros(1, 7)}) == []


def test_goal_pose_config_validation_and_scheduler_group():
    with pytest.raises(ValueError):
        # se3_encoder source needs a target_pose_delta_index.
        MolmoAct2Config(
            checkpoint_path="/tmp/x",
            action_mode="continuous",
            enable_goal_pose=True,
            goal_token_source="se3_encoder",
        )
    with pytest.raises(ValueError):
        # mask_image / pose reconstruction require enable_goal_pose.
        MolmoAct2Config(checkpoint_path="/tmp/x", mask_image_from_action_expert=True)

    cfg = MolmoAct2Config(
        checkpoint_path="/tmp/x",
        action_mode="continuous",
        enable_goal_pose=True,
        goal_token_source="learnable_queries",
        num_goal_tokens=4,
        target_pose_delta_index=10,
        enable_pose_reconstruction=True,
        mask_image_from_action_expert=True,
        scheduler_goal_warmup_steps=500,
    )
    assert cfg.enable_goal_pose
    assert cfg.get_scheduler_preset().goal_warmup_steps == 500
    assert cfg.target_pose_delta_index == 10


def test_semantic_visual_config_validation_and_scheduler_group():
    cfg = MolmoAct2Config(
        checkpoint_path="/tmp/x",
        action_mode="continuous",
        enable_goal_pose=True,
        goal_token_source="learnable_queries",
        goal_conditioning_mode="semantic_visual_recurrent",
        target_pose_delta_index=10,
        enable_pose_reconstruction=True,
        mask_image_from_action_expert=True,
        num_semantic_visual_tokens=100,
        semantic_visual_hidden_dim=768,
        semantic_visual_num_heads=8,
        scheduler_semantic_visual_warmup_steps=1000,
    )
    assert cfg.inferred_max_sequence_length() == MolmoAct2Config(
        checkpoint_path="/tmp/x",
        action_mode="continuous",
    ).inferred_max_sequence_length()
    assert cfg.get_scheduler_preset().semantic_visual_warmup_steps == 1000

    with pytest.raises(ValueError):
        MolmoAct2Config(
            checkpoint_path="/tmp/x",
            action_mode="continuous",
            enable_goal_pose=True,
            goal_token_source="learnable_queries",
            goal_conditioning_mode="semantic_visual_recurrent",
            target_pose_delta_index=10,
            enable_pose_reconstruction=True,
            mask_image_from_action_expert=False,
        )


def test_infer_max_sequence_length_accounts_for_goal_tokens():
    base = infer_molmoact2_max_sequence_length(
        num_images=2, state_dim=8, action_dim=7, action_horizon=10, include_discrete_action=False
    )
    with_goal = infer_molmoact2_max_sequence_length(
        num_images=2,
        state_dim=8,
        action_dim=7,
        action_horizon=10,
        include_discrete_action=False,
        num_goal_tokens=64,
    )
    assert with_goal >= base


def test_goal_se3_encoder_and_decoder_shapes():
    encoder = molmoact2_modeling._GoalSE3Encoder(pose_dim=8, num_tokens=4, hidden_size=16, inner_dim=32)
    decoder = molmoact2_modeling._GoalPoseDecoder(num_tokens=4, hidden_size=16, pose_dim=8, inner_dim=32)

    tokens = encoder(torch.randn(3, 8))
    assert tokens.shape == (3, 4, 16)
    assert decoder(tokens).shape == (3, 8)


def test_semantic_visual_aggregator_recurrent_shapes_and_pose_decoder():
    aggregator = molmoact2_modeling._SemanticVisualAggregator(
        num_tokens=100,
        latent_dim=32,
        context_dim=48,
        kv_dim=24,
        num_heads=4,
        ffn_ratio=2.0,
        dropout=0.0,
    )
    layer_hidden = torch.randn(2, 12, 48)
    semantic_mask = torch.tensor(
        [[True] * 5 + [False] * 7, [True] * 5 + [False] * 7]
    )
    image_mask = ~semantic_mask
    q0 = aggregator.initial_queries(2, device=torch.device("cpu"), dtype=torch.float32)
    q1 = aggregator(q0, layer_hidden, semantic_mask=semantic_mask, image_mask=image_mask)
    q2 = aggregator(q1, layer_hidden, semantic_mask=semantic_mask, image_mask=image_mask)
    key, value = aggregator.project_kv(q2)

    assert q0.shape == q1.shape == q2.shape == (2, 100, 32)
    assert not torch.equal(q1, q2)
    assert key.shape == value.shape == (2, 100, 24)
    decoder = molmoact2_modeling._SemanticVisualPoseDecoder(32, 8, 16)
    assert decoder(q2).shape == (2, 8)


def test_goal_pose_modules_are_built_only_for_the_active_stage():
    def make_policy(config):
        policy = object.__new__(MolmoAct2Policy)
        torch.nn.Module.__init__(policy)
        policy.config = config
        policy._resolve_backbone_hidden_size = lambda: 16
        policy._goal_pose_dim = lambda: 8
        policy._action_expert = lambda: SimpleNamespace(llm_kv_dim=12)
        policy._build_goal_pose_modules(torch.float32)
        return policy

    stage1 = make_policy(
        SimpleNamespace(
            enable_goal_pose=True,
            goal_conditioning_mode="vlm_appended",
            goal_token_source="se3_encoder",
            goal_hidden_dim=32,
            num_goal_tokens=4,
        )
    )
    assert hasattr(stage1, "goal_se3_encoder")
    assert not hasattr(stage1, "goal_queries")
    assert not hasattr(stage1, "goal_pose_decoder")

    v1_stage2 = make_policy(
        SimpleNamespace(
            enable_goal_pose=True,
            goal_conditioning_mode="vlm_appended",
            goal_token_source="learnable_queries",
            goal_hidden_dim=32,
            num_goal_tokens=4,
            init_queries_from_se3_encoder=False,
            enable_pose_reconstruction=True,
        )
    )
    assert hasattr(v1_stage2, "goal_queries")
    assert hasattr(v1_stage2, "goal_pose_decoder")
    assert not hasattr(v1_stage2, "goal_se3_encoder")

    v2_stage2 = make_policy(
        SimpleNamespace(
            enable_goal_pose=True,
            goal_conditioning_mode="semantic_visual_recurrent",
            goal_token_source="learnable_queries",
            goal_hidden_dim=32,
            num_semantic_visual_tokens=10,
            semantic_visual_hidden_dim=16,
            semantic_visual_num_heads=4,
            semantic_visual_ffn_ratio=2.0,
            semantic_visual_dropout=0.0,
        )
    )
    assert hasattr(v2_stage2, "semantic_visual_aggregator")
    assert hasattr(v2_stage2, "semantic_visual_pose_decoder")
    assert not hasattr(v2_stage2, "goal_se3_encoder")
    assert not hasattr(v2_stage2, "goal_queries")
    assert not hasattr(v2_stage2, "goal_pose_decoder")


def test_goal_token_embeddings_from_queries_and_encoder():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(enable_goal_pose=True, goal_token_source="learnable_queries")
    policy.goal_queries = torch.nn.Parameter(torch.randn(4, 16))
    queries = policy._goal_token_embeddings(
        {}, batch_size=3, device=torch.device("cpu"), dtype=torch.float32
    )
    assert queries.shape == (3, 4, 16)

    policy.config = SimpleNamespace(enable_goal_pose=True, goal_token_source="se3_encoder")
    policy.goal_se3_encoder = molmoact2_modeling._GoalSE3Encoder(8, 4, 16, 32)
    with pytest.raises(ValueError):
        policy._goal_token_embeddings({}, batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
    encoded = policy._goal_token_embeddings(
        {"goal_pose": torch.randn(2, 8)}, batch_size=2, device=torch.device("cpu"), dtype=torch.float32
    )
    assert encoded.shape == (2, 4, 16)


def test_encoder_attention_mask_masks_image_and_appends_goal_tokens():
    policy = object.__new__(MolmoAct2Policy)
    policy.config = SimpleNamespace(action_mode="continuous", mask_image_from_action_expert=True)
    policy.model = SimpleNamespace(
        config=SimpleNamespace(image_patch_id=999, image_low_res_id=None, eos_token_id=None)
    )
    backbone = SimpleNamespace(config=SimpleNamespace())
    policy._backbone = lambda: backbone

    input_ids = torch.tensor([[5, 999, 999, 7]])
    attention_mask = torch.ones(1, 4, dtype=torch.bool)
    mask = policy._encoder_attention_mask_for_action_expert(
        input_ids=input_ids, attention_mask=attention_mask, num_goal_tokens=2
    )
    assert mask.shape == (1, 6)
    # image-patch positions hidden from the action expert, goal columns always visible.
    assert mask[0].tolist() == [True, False, False, True, True, True]


def test_semantic_visual_masks_split_semantic_and_image_feature_tokens():
    policy = object.__new__(MolmoAct2Policy)
    policy.model = SimpleNamespace(
        config=SimpleNamespace(image_patch_id=999, image_low_res_id=998)
    )
    policy._backbone = lambda: SimpleNamespace(config=SimpleNamespace())
    input_ids = torch.tensor([[10, 999, 999, 11, 998, 12]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    token_type_ids = torch.tensor([[False, True, True, False, True, False]])

    semantic, image = policy._semantic_visual_context_masks(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    assert semantic[0].tolist() == [True, False, False, True, False, True]
    assert image[0].tolist() == [False, True, True, False, True, False]
    augmented = policy._append_context_columns(semantic, 100)
    assert augmented.shape == (1, 106)
    assert bool(augmented[:, -100:].all())


def test_pose_reconstruction_loss_honors_pad_mask():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        enable_goal_pose=True, enable_pose_reconstruction=True, num_goal_tokens=4
    )
    policy.goal_pose_decoder = molmoact2_modeling._GoalPoseDecoder(4, 16, 8, 32)
    hidden = torch.randn(2, 10, 16)
    batch = {
        "goal_pose": torch.randn(2, 8),
        "goal_pose_is_pad": torch.tensor([False, True]),
    }
    loss = policy._compute_pose_reconstruction_loss(batch, hidden)
    assert loss is not None and loss.ndim == 0

    policy.config = SimpleNamespace(
        enable_goal_pose=True, enable_pose_reconstruction=False, num_goal_tokens=4
    )
    assert policy._compute_pose_reconstruction_loss(batch, hidden) is None


def test_semantic_visual_pose_reconstruction_uses_all_tokens_and_pad_mask():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        enable_goal_pose=True,
        enable_pose_reconstruction=True,
        goal_conditioning_mode="semantic_visual_recurrent",
        num_semantic_visual_tokens=100,
    )
    policy.semantic_visual_pose_decoder = molmoact2_modeling._SemanticVisualPoseDecoder(32, 8, 16)
    tokens = torch.randn(2, 100, 32)
    batch = {
        "goal_pose": torch.randn(2, 8),
        "goal_pose_is_pad": torch.tensor([False, True]),
    }
    loss = policy._compute_pose_reconstruction_loss(batch, tokens)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)


def test_extract_state_and_goal_pose_from_multi_frame_state():
    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.enable_goal_pose = True
    observation = {OBS_STATE: torch.randn(2, 3, 8)}

    current = step._extract_state(observation, 2)
    assert current.shape == (2, 8)
    assert torch.equal(current, observation[OBS_STATE][:, 0])

    complementary = {
        "observation.state_is_pad": torch.tensor([[False, False, True], [False, False, False]])
    }
    goal_pose, goal_pose_is_pad = step._extract_goal_pose(observation, complementary, 2)
    assert goal_pose.shape == (2, 8)
    assert torch.equal(goal_pose, observation[OBS_STATE][:, -1])
    assert goal_pose_is_pad.tolist() == [True, False]

    # Single-frame state (inference) has no target frame.
    assert step._extract_goal_pose({OBS_STATE: torch.randn(2, 8)}, {}, 2) == (None, None)


def test_resolve_delta_timestamps_adds_target_state_frame_only():
    from lerobot.datasets.factory import resolve_delta_timestamps

    cfg = SimpleNamespace(
        reward_delta_indices=None,
        action_delta_indices=[0, 1, 2],
        observation_delta_indices=None,
        target_pose_delta_index=10,
    )
    ds_meta = SimpleNamespace(
        features={"observation.state": {}, "observation.images.image": {}, "action": {}},
        fps=10.0,
    )
    delta = resolve_delta_timestamps(cfg, ds_meta)
    assert delta["action"] == pytest.approx([0.0, 0.1, 0.2])
    assert delta["observation.state"] == pytest.approx([0.0, 1.0])
    # Cameras stay single-frame (no future image loading).
    assert "observation.images.image" not in delta


def test_molmoact2_rollout_generator_uses_eval_seed_per_task():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(per_episode_seed=True, eval_seed=1000)
    policy._rollout_action_generator = None
    policy._rollout_task_key = None
    policy._rollout_index_for_task = -1

    policy.reset()
    first = policy._rollout_generator_for_inputs(
        {"task": ["pick", "pick", "pick"]},
        batch_size=3,
        device=torch.device("cpu"),
    )
    expected_first = torch.Generator().manual_seed(
        MolmoAct2Policy._combine_rollout_seeds(first_seed=1000, batch_size=3)
    )
    assert torch.allclose(torch.rand(4, generator=first), torch.rand(4, generator=expected_first))

    policy.reset()
    second = policy._rollout_generator_for_inputs(
        {"task": ["pick", "pick", "pick"]},
        batch_size=3,
        device=torch.device("cpu"),
    )
    expected_second = torch.Generator().manual_seed(
        MolmoAct2Policy._combine_rollout_seeds(first_seed=1003, batch_size=3)
    )
    assert torch.allclose(torch.rand(4, generator=second), torch.rand(4, generator=expected_second))

    policy.reset()
    new_task = policy._rollout_generator_for_inputs(
        {"task": ["place", "place", "place"]},
        batch_size=3,
        device=torch.device("cpu"),
    )
    expected_new_task = torch.Generator().manual_seed(
        MolmoAct2Policy._combine_rollout_seeds(first_seed=1000, batch_size=3)
    )
    assert torch.allclose(torch.rand(4, generator=new_task), torch.rand(4, generator=expected_new_task))


def test_molmoact2_gripper_mask_uses_feature_names(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "features": {
                    ACTION: {"names": {"motors": ["x", "gripper"]}},
                    OBS_STATE: {"names": {"motors": ["joint", "gripper"]}},
                }
            }
        ),
        encoding="utf-8",
    )
    dataset_meta = SimpleNamespace(root=tmp_path)
    stats = {
        ACTION: {"q01": [0.0, 0.0], "q99": [10.0, 10.0]},
        OBS_STATE: {"q01": [0.0, 0.0], "q99": [10.0, 10.0]},
    }

    masked_stats = _add_gripper_masks_to_stats(stats, dataset_meta, normalize_gripper=False)

    assert masked_stats is not None
    assert masked_stats[ACTION]["mask"] == [True, False]
    assert masked_stats[OBS_STATE]["mask"] == [True, False]

    features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
    }
    norm_map = {
        FeatureType.ACTION: NormalizationMode.QUANTILES,
        FeatureType.STATE: NormalizationMode.QUANTILES,
    }
    transition = {
        TransitionKey.OBSERVATION: {OBS_STATE: torch.tensor([[5.0, 7.0]])},
        TransitionKey.ACTION: torch.tensor([[5.0, 7.0]]),
    }
    normalizer = MolmoAct2MaskedNormalizerProcessorStep(
        features=features,
        norm_map=norm_map,
        stats=masked_stats,
    )
    normalized = normalizer(transition)

    assert torch.equal(normalized[TransitionKey.OBSERVATION][OBS_STATE], torch.tensor([[0.0, 7.0]]))
    assert torch.equal(normalized[TransitionKey.ACTION], torch.tensor([[0.0, 7.0]]))

    unnormalizer = MolmoAct2MaskedUnnormalizerProcessorStep(
        features={ACTION: features[ACTION]},
        norm_map=norm_map,
        stats=masked_stats,
    )
    unnormalized = unnormalizer({TransitionKey.ACTION: torch.tensor([[0.0, 7.0]])})

    assert torch.equal(unnormalized[TransitionKey.ACTION], torch.tensor([[5.0, 7.0]]))


def test_molmoact2_normalize_gripper_true_keeps_all_dims_normalized(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps({"features": {ACTION: {"names": ["x", "gripper"]}}}),
        encoding="utf-8",
    )
    stats = {ACTION: {"q01": [0.0, 0.0], "q99": [10.0, 10.0]}}

    masked_stats = _add_gripper_masks_to_stats(
        stats,
        SimpleNamespace(root=tmp_path),
        normalize_gripper=True,
    )

    assert masked_stats is not None
    assert masked_stats[ACTION]["mask"] == [True, True]


def test_molmoact2_uses_supplied_stats_with_repo_scoped_names(tmp_path):
    repo_root = tmp_path / "test-org" / "libero"
    (repo_root / "meta").mkdir(parents=True)
    (repo_root / "meta" / "info.json").write_text(
        json.dumps({"features": {ACTION: {"names": ["x", "gripper"]}}}),
        encoding="utf-8",
    )
    base_stats = {ACTION: {"q01": [0.0, 0.0], "q99": [10.0, 10.0]}}

    masked_stats = _add_gripper_masks_to_stats(
        base_stats,
        SimpleNamespace(root=tmp_path, repo_id="test-org/libero"),
        normalize_gripper=False,
    )

    assert masked_stats is not None
    assert masked_stats[ACTION]["q01"] == [0.0, 0.0]
    assert masked_stats[ACTION]["mask"] == [True, False]


def test_molmoact2_uses_config_feature_names_without_dataset_meta():
    base_stats = {ACTION: {"q01": [0.0, 0.0], "q99": [10.0, 10.0]}}

    masked_stats = _add_gripper_masks_to_stats(
        base_stats,
        None,
        normalize_gripper=False,
        dataset_feature_names={ACTION: ["x", "gripper"]},
    )

    assert masked_stats is not None
    assert masked_stats[ACTION]["mask"] == [True, False]


def test_enable_lora_vlm_builds_policy_local_peft_config():
    pytest.importorskip("peft")
    policy_cfg = MolmoAct2Config(
        checkpoint_path="/tmp/not-a-real-checkpoint",
        device="cpu",
        enable_lora_vlm=True,
        lora_rank=64,
        push_to_hub=False,
    )
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = policy_cfg

    peft_config = policy._build_inner_lora_config()

    assert peft_config.r == 64
    assert peft_config.target_modules == policy._get_inner_peft_targets()["target_modules"]
    assert not policy_cfg.use_peft


def test_cuda_graph_managers_are_inference_only():
    class DummyManager:
        def __init__(self):
            self.enabled = None

        def set_enabled(self, enabled):
            self.enabled = enabled

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_cuda_graph_manager = DummyManager()

        def _require_action_expert(self):
            return torch.nn.Linear(1, 1)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyBackbone()
            self.depth_decode_cuda_graph_manager = DummyManager()

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(train_action_expert_only=False, enable_inference_cuda_graph=True)
    policy.model = DummyModel()

    policy.train()
    assert policy.model.model.action_cuda_graph_manager.enabled is False
    assert policy.model.depth_decode_cuda_graph_manager.enabled is False

    policy.eval()
    assert policy.model.model.action_cuda_graph_manager.enabled is True
    assert policy.model.depth_decode_cuda_graph_manager.enabled is True

    policy.config.enable_inference_cuda_graph = False
    policy.eval()
    assert policy.model.model.action_cuda_graph_manager.enabled is False
    assert policy.model.depth_decode_cuda_graph_manager.enabled is False


def test_lora_action_expert_target_is_opt_in():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        lora_rank=64,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_bias="none",
        enable_lora_action_expert=False,
    )

    targets = policy._get_default_peft_targets()["target_modules"]

    assert "transformer|vision_backbone" in targets
    assert "action_expert" not in targets

    policy.config.enable_lora_action_expert = True
    targets = policy._get_default_peft_targets()["target_modules"]

    assert "action_expert" in targets
    assert "state_encoder" not in targets
    assert "state_norm" not in targets
    assert "kv_proj" not in targets


def test_enable_lora_vlm_wraps_loaded_hf_model_locally():
    pytest.importorskip("peft")

    class DummyInnerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.wq = torch.nn.Linear(2, 2)
            self.action_expert = torch.nn.Module()
            self.action_expert.action_embed = torch.nn.Linear(2, 2)

    class DummyHFModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {}
            self.model = DummyInnerModel()

        def forward(self, x):
            return self.model.transformer.wq(x)

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        checkpoint_path="/tmp/base",
        lora_rank=2,
        lora_alpha=4,
        lora_dropout=0.0,
        lora_bias="none",
        enable_lora_action_expert=False,
        train_action_expert_only=False,
        enable_inference_cuda_graph=False,
    )
    policy.model = DummyHFModel()

    policy._apply_lora_adapters()

    assert policy._backbone() is policy.model.base_model.model.model
    trainable = [name for name, param in policy.named_parameters() if param.requires_grad]
    assert trainable
    assert any("lora_" in name for name in trainable)
    assert any("action_expert.action_embed" in name and "lora_" not in name for name in trainable)
    assert policy.model(torch.ones(1, 2)).shape == (1, 2)


def test_lora_vlm_unfreezes_action_expert_base_weights():
    class DummyInnerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.wq = torch.nn.Linear(2, 2)
            self.action_expert = torch.nn.Module()
            self.action_expert.action_embed = torch.nn.Linear(2, 2)

    class DummyHFModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyInnerModel()

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.model = DummyHFModel()

    for param in policy.parameters():
        param.requires_grad_(False)
    policy._unfreeze_action_expert_parameters()

    trainable = [name for name, param in policy.named_parameters() if param.requires_grad]
    assert trainable
    assert all("action_expert" in name for name in trainable)


def test_train_action_expert_only_requires_continuous_action_mode():
    with pytest.raises(ValueError, match="requires action_mode='continuous'"):
        MolmoAct2Config(action_mode="both", train_action_expert_only=True)

    with pytest.raises(ValueError, match="incompatible with enable_lora_vlm"):
        MolmoAct2Config(action_mode="continuous", train_action_expert_only=True, enable_lora_vlm=True)

    cfg = MolmoAct2Config(action_mode="continuous", train_action_expert_only=True)
    assert cfg.train_action_expert_only


def test_molmoact2_sequence_length_is_inferred_from_fixed_token_budget():
    cfg = MolmoAct2Config(
        action_mode="both",
        chunk_size=10,
        n_action_steps=10,
        image_keys=["observation.images.image", "observation.images.wrist_image"],
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(8,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
    )

    assert cfg.max_sequence_length is None
    assert cfg.inferred_max_sequence_length() == 640
    assert cfg.inferred_max_sequence_length(include_discrete_action=False) == 576
    assert (
        infer_molmoact2_max_sequence_length(
            num_images=2,
            state_dim=8,
            action_dim=7,
            action_horizon=30,
            include_discrete_action=True,
        )
        == 768
    )


def test_molmoact2_sequence_length_override_is_preserved():
    cfg = MolmoAct2Config(max_sequence_length=1024)

    assert cfg.inferred_max_sequence_length(num_images=2, state_dim=8, action_dim=7) == 1024


def test_train_action_expert_only_freezes_non_action_expert_params():
    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Linear(2, 2)
            self.vision_backbone = torch.nn.Linear(2, 2)
            self.action_expert = torch.nn.Linear(2, 2)

        def _require_action_expert(self):
            return self.action_expert

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = DummyBackbone()
            self.lm_head = torch.nn.Linear(2, 2)

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(train_action_expert_only=True)
    policy.model = DummyModel()

    policy._freeze_non_action_expert_parameters()
    policy.train()

    assert policy.model.model.action_expert.training
    assert not policy.model.training
    assert not policy.model.model.transformer.training
    assert all(param.requires_grad for param in policy.model.model.action_expert.parameters())
    assert not any(param.requires_grad for param in policy.model.model.transformer.parameters())
    assert not any(param.requires_grad for param in policy.model.model.vision_backbone.parameters())
    assert not any(param.requires_grad for param in policy.model.lm_head.parameters())


def test_load_hf_model_accepts_max_action_horizon_schema(monkeypatch):
    class DummyLoadedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                max_action_dim=32,
                max_action_horizon=30,
                action_mode="both",
                add_action_expert=True,
            )
            self.model = torch.nn.Module()
            self.embed_tokens = torch.nn.Embedding(4, 4)
            self.lm_head = torch.nn.Linear(4, 4, bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

    loaded_model = DummyLoadedModel()
    resolved_kwargs = {}

    def fake_resolve_checkpoint_location(checkpoint_path, **kwargs):
        resolved_kwargs.update(kwargs)
        return checkpoint_path

    monkeypatch.setattr(molmoact2_modeling, "_resolve_checkpoint_location", fake_resolve_checkpoint_location)
    monkeypatch.setattr(molmoact2_modeling, "_patch_batched_image_attention_bias", lambda backbone: None)
    monkeypatch.setattr(molmoact2_modeling, "_patch_leaf_safe_input_embedding_update", lambda backbone: None)
    monkeypatch.setattr(molmoact2_modeling, "_patch_training_kv_collection", lambda backbone: None)

    from transformers import AutoModelForImageTextToText

    monkeypatch.setattr(
        AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: loaded_model,
    )
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(
        checkpoint_path="/tmp/new-schema-checkpoint",
        checkpoint_revision="main",
        checkpoint_force_download=True,
        chunk_size=10,
        n_action_steps=10,
        action_mode="both",
    )

    policy._load_hf_model()

    assert policy.model is loaded_model
    assert not hasattr(policy.model.config, "action_horizon")
    assert policy.model.config.max_action_horizon == 10
    assert policy._generation_action_horizon() == 10
    assert resolved_kwargs == {"revision": "main", "force_download": True}


def test_load_hf_model_chunk_size_overrides_larger_than_checkpoint_horizon(monkeypatch):
    class DummyLoadedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                max_action_dim=32,
                max_action_horizon=10,
                action_mode="both",
                add_action_expert=True,
            )
            self.model = torch.nn.Module()
            self.embed_tokens = torch.nn.Embedding(4, 4)
            self.lm_head = torch.nn.Linear(4, 4, bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

    loaded_model = DummyLoadedModel()
    monkeypatch.setattr(
        molmoact2_modeling,
        "_resolve_checkpoint_location",
        lambda checkpoint_path, **kwargs: checkpoint_path,
    )
    monkeypatch.setattr(molmoact2_modeling, "_patch_batched_image_attention_bias", lambda backbone: None)
    monkeypatch.setattr(molmoact2_modeling, "_patch_leaf_safe_input_embedding_update", lambda backbone: None)
    monkeypatch.setattr(molmoact2_modeling, "_patch_training_kv_collection", lambda backbone: None)

    from transformers import AutoModelForImageTextToText

    monkeypatch.setattr(
        AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: loaded_model,
    )
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(
        checkpoint_path="/tmp/new-schema-checkpoint",
        chunk_size=30,
        n_action_steps=30,
        action_mode="both",
    )

    policy._load_hf_model()

    assert policy.model.config.max_action_horizon == 30
    assert policy._generation_action_horizon() == 30


def test_load_hf_model_rejects_legacy_action_horizon_schema(monkeypatch):
    class DummyLoadedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                max_action_dim=32,
                action_horizon=30,
                action_mode="both",
                add_action_expert=True,
            )
            self.model = torch.nn.Module()

    monkeypatch.setattr(
        molmoact2_modeling,
        "_resolve_checkpoint_location",
        lambda checkpoint_path, **kwargs: checkpoint_path,
    )

    from transformers import AutoModelForImageTextToText

    monkeypatch.setattr(
        AutoModelForImageTextToText,
        "from_pretrained",
        lambda *args, **kwargs: DummyLoadedModel(),
    )
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = MolmoAct2Config(
        checkpoint_path="/tmp/legacy-schema-checkpoint",
        chunk_size=10,
        n_action_steps=10,
        action_mode="both",
    )

    with pytest.raises(ValueError, match="max_action_horizon"):
        policy._load_hf_model()


def test_rtc_processor_initialization_and_select_action_guard():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(rtc_config=RTCConfig(enabled=True))

    policy.init_rtc_processor()

    assert policy.rtc_processor is not None
    with pytest.raises(AssertionError, match="RTC is not supported for select_action"):
        policy.select_action({})


def test_inference_action_mode_is_explicit_and_has_no_action_mode_alias():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(action_mode="both", inference_action_mode=None)
    policy._checkpoint_action_mode = None

    with pytest.raises(ValueError, match="inference_action_mode.*explicitly"):
        policy._resolve_inference_action_mode(None)
    with pytest.raises(TypeError, match="unexpected keyword argument 'action_mode'"):
        policy.predict_action_chunk({}, action_mode="continuous")


def test_rtc_generation_uses_previous_chunk_prefix():
    class DummyActionExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def prepare_context(self, **kwargs):
            del kwargs
            return SimpleNamespace()

        def get_or_prepare_modulation_cache(self, timesteps, *, cache_key=None):
            del cache_key
            return [SimpleNamespace(conditioning=timestep) for timestep in timesteps]

        def forward_with_context(self, actions, timesteps, *, context, modulation=None):
            del timesteps, context, modulation
            return torch.ones_like(actions) * self.weight

    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                flow_matching_num_steps=2,
                max_action_horizon=4,
                max_action_dim=3,
            )
            self.action_expert = DummyActionExpert()
            self.batch_size = 1

        def _require_action_expert(self):
            return self.action_expert

        def forward(self, **kwargs):
            self.batch_size = int(kwargs["input_ids"].shape[0])
            return SimpleNamespace(past_key_values=object())

        def _extract_kv_states(self, past_key_values):
            del past_key_values
            kv = torch.zeros(self.batch_size, 1, 1)
            return [(kv, kv)]

        def _get_encoder_attention_mask(self, input_ids, attention_mask):
            del input_ids
            return attention_mask

        def _depth_gate_from_condition(self, **kwargs):
            del kwargs
            return None, None

        def _apply_depth_gate_to_layer_kv_states(self, encoder_kv_states, depth_mask, depth_gate):
            del depth_mask, depth_gate
            return encoder_kv_states

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        mask_action_dim_padding=True,
        rtc_config=RTCConfig(enabled=True, execution_horizon=2, max_guidance_weight=1.0),
    )
    policy.rtc_processor = None
    policy.model = torch.nn.Module()
    policy.model.model = DummyBackbone()
    policy.init_rtc_processor()
    model_inputs = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    action_dim_is_pad = torch.tensor([[False, False, False]])

    without_prefix = policy._generate_actions_from_inputs_with_rtc(
        model_inputs=model_inputs,
        action_dim_is_pad=action_dim_is_pad,
        num_steps=2,
        generator=torch.Generator().manual_seed(0),
        inference_delay=0,
        prev_chunk_left_over=None,
        execution_horizon=None,
    )
    with_prefix = policy._generate_actions_from_inputs_with_rtc(
        model_inputs=model_inputs,
        action_dim_is_pad=action_dim_is_pad,
        num_steps=2,
        generator=torch.Generator().manual_seed(0),
        inference_delay=0,
        prev_chunk_left_over=torch.zeros(1, 4, 3),
        execution_horizon=None,
    )

    assert without_prefix.shape == (1, 4, 3)
    assert not torch.allclose(without_prefix, with_prefix)


def test_discrete_state_string_matches_molmoact2_bins():
    state = np.asarray([-1.0, 0.0, 1.0, np.nan, np.inf, -np.inf], dtype=np.float32)

    assert _build_discrete_state_string(state, 256) == (
        "<state_start><state_0><state_128><state_255><state_128><state_255><state_0><state_end>"
    )


def test_question_normalization_matches_release_prompt_style():
    assert _normalize_question_text("Instruction: Pick up the cube, please!") == "pick up the cube, please"
    assert (
        _normalize_question_text("The task is to open drawer. Then close it.") == "open drawer; then close it"
    )


def test_action_padding_marks_only_real_dimensions():
    step = object.__new__(MolmoAct2PackInputsProcessorStep)
    step.max_action_dim = 32
    action = torch.ones(2, 3, 7)

    padded, horizon_is_pad, dim_is_pad = step._pad_action(action, None)

    assert padded.shape == (2, 3, 32)
    assert torch.equal(padded[..., :7], action)
    assert torch.count_nonzero(padded[..., 7:]) == 0
    assert not horizon_is_pad.any()
    assert not dim_is_pad[:, :7].any()
    assert dim_is_pad[:, 7:].all()


def test_action_dim_padding_loss_reduces_like_old_trainer():
    loss = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)
    action_dim_is_pad = torch.tensor(
        [
            [False, False, True, True],
            [False, True, True, True],
        ]
    )

    reduced = MolmoAct2Policy._apply_action_dim_padding_mask(loss, action_dim_is_pad)

    expected = torch.stack(
        [
            loss[0, :, :, :2].sum(dim=-1) / 2,
            loss[1, :, :, :1].sum(dim=-1) / 1,
        ],
        dim=0,
    )
    assert torch.equal(reduced, expected)


def test_action_chunk_padding_keeps_old_mean_denominator():
    loss = torch.ones(1, 2, 4, 3)
    action_horizon_is_pad = torch.tensor([[False, False, True, True]])

    masked = MolmoAct2Policy._apply_action_chunk_padding_mask(loss, action_horizon_is_pad)

    assert masked.mean().item() == 0.5


def test_selected_discrete_loss_matches_full_causal_lm_loss():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        softmax_auxiliary_loss=False,
        softmax_auxiliary_loss_scale=1e-4,
        discrete_loss_token_weighting="none",
    )
    policy.model = torch.nn.Module()
    policy.model.lm_head = torch.nn.Linear(3, 5, bias=False)
    outputs = type("Outputs", (), {})()
    outputs.last_hidden_state = torch.randn(2, 4, 3)
    labels = torch.tensor(
        [
            [-100, 1, 2, -100],
            [-100, -100, 3, 4],
        ]
    )

    selected_loss, z_loss = policy._discrete_loss_from_backbone_outputs({"labels": labels}, outputs)

    logits = policy.model.lm_head(outputs.last_hidden_state)
    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    expected_loss = F.cross_entropy(logits.float().view(-1, 5), shift_labels.view(-1), ignore_index=-100)
    assert torch.allclose(selected_loss, expected_loss)
    assert z_loss is None


def test_discrete_z_loss_matches_old_trainer_formula():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        discrete_loss_token_weighting="none",
    )
    policy.model = torch.nn.Module()
    policy.model.lm_head = torch.nn.Linear(3, 5, bias=False)
    outputs = type("Outputs", (), {})()
    outputs.last_hidden_state = torch.randn(2, 4, 3)
    labels = torch.tensor(
        [
            [-100, 1, 2, -100],
            [-100, -100, 3, 4],
        ]
    )

    ce_loss, z_loss = policy._discrete_loss_from_backbone_outputs({"labels": labels}, outputs)

    logits = policy.model.lm_head(outputs.last_hidden_state).float()
    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    valid = shift_labels != -100
    expected_ce = F.cross_entropy(logits.view(-1, 5), shift_labels.view(-1), ignore_index=-100)
    expected_z = 1e-4 * logits.logsumexp(dim=-1)[valid].pow(2).mean()
    assert torch.allclose(ce_loss, expected_ce)
    assert z_loss is not None
    assert torch.allclose(z_loss, expected_z)


def test_discrete_reduction_none_preserves_mean_loss():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        discrete_loss_token_weighting="root_subsegments_root_tokens",
    )
    policy.model = torch.nn.Module()
    policy.model.lm_head = torch.nn.Linear(3, 5, bias=False)
    outputs = type("Outputs", (), {})()
    outputs.last_hidden_state = torch.randn(3, 5, 3)
    labels = torch.tensor(
        [
            [-100, 1, -100, -100, -100],
            [-100, -100, 2, 3, -100],
            [-100, 4, 3, 2, 1],
        ]
    )

    ce_mean, z_mean = policy._discrete_loss_from_backbone_outputs(
        {"labels": labels},
        outputs,
        reduction="mean",
    )
    ce_none, z_none = policy._discrete_loss_from_backbone_outputs(
        {"labels": labels},
        outputs,
        reduction="none",
    )

    assert ce_none.shape == (3,)
    assert z_none is not None
    assert z_none.shape == (3,)
    assert torch.allclose(ce_none.mean(), ce_mean)
    assert torch.allclose(z_none.mean(), z_mean)


def test_forward_reduction_none_returns_per_sample_discrete_loss():
    class DummyBackbone(torch.nn.Module):
        def __init__(self, hidden_states):
            super().__init__()
            self.hidden_states = hidden_states

        def forward(self, **kwargs):
            del kwargs
            return SimpleNamespace(last_hidden_state=self.hidden_states)

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        action_mode="discrete",
        inference_action_mode="discrete",
        model_dtype="float32",
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        discrete_loss_token_weighting="none",
    )
    policy.model = torch.nn.Module()
    policy.model.lm_head = torch.nn.Linear(3, 5, bias=False)
    hidden_states = torch.randn(2, 4, 3)
    policy._backbone = lambda: DummyBackbone(hidden_states)
    batch = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.tensor(
            [
                [-100, 1, 2, -100],
                [-100, -100, 3, 4],
            ]
        ),
    }

    loss_none, metrics_none = policy.forward(batch, reduction="none")
    loss_mean, metrics_mean = policy.forward(batch, reduction="mean")

    assert loss_none.shape == (2,)
    assert torch.allclose(loss_none.mean(), loss_mean)
    assert metrics_none["loss"] == pytest.approx(metrics_mean["loss"])


def test_discrete_root_token_weighting_matches_old_loss_mask_scaling():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        softmax_auxiliary_loss=True,
        softmax_auxiliary_loss_scale=1e-4,
        discrete_loss_token_weighting="root_subsegments_root_tokens",
    )
    policy.model = torch.nn.Module()
    policy.model.lm_head = torch.nn.Linear(3, 5, bias=False)
    outputs = type("Outputs", (), {})()
    outputs.last_hidden_state = torch.randn(2, 4, 3)
    labels = torch.tensor(
        [
            [-100, -100, 1, -100],
            [-100, 2, 3, 4],
        ]
    )

    ce_loss, z_loss = policy._discrete_loss_from_backbone_outputs({"labels": labels}, outputs)

    logits = policy.model.lm_head(outputs.last_hidden_state).float()
    shift_labels = F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
    valid = shift_labels != -100
    log_z = logits.logsumexp(dim=-1)
    token_ce = log_z - logits.gather(dim=-1, index=shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    weights = torch.zeros_like(token_ce)
    counts = valid.sum(dim=1).float()
    weights[valid] = (2.0 / torch.sqrt(counts))[:, None].expand_as(weights)[valid]
    expected_ce = (token_ce * weights).sum() / weights.sum()
    expected_z = 1e-4 * (log_z.pow(2) * weights).sum() / weights.sum()
    assert torch.allclose(ce_loss, expected_ce)
    assert z_loss is not None
    assert torch.allclose(z_loss, expected_z)


class _DummyActionTokenizer:
    def decode(self, tokens, *, time_horizon=None, action_dim=None):
        decoded = []
        for token_row in tokens:
            decoded.append(np.full((time_horizon, action_dim), sum(token_row), dtype=np.float32))
        return np.stack(decoded)


def test_discrete_decode_extracts_action_bins_for_each_batch():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(chunk_size=2)
    policy.action_tokenizer = _DummyActionTokenizer()
    policy.model = torch.nn.Module()
    policy.model.config = SimpleNamespace(
        action_start_token_id=10,
        action_end_token_id=11,
        action_token_start_id=100,
        num_action_tokens=4,
        action_horizon=2,
    )

    actions = policy._decode_discrete_action_chunk(
        torch.tensor(
            [
                [10, 100, 101, 11, 2],
                [10, 102, 103, 11, 2],
            ]
        ),
        action_dim=2,
    )

    assert actions.shape == (2, 2, 2)
    assert torch.equal(actions[0], torch.ones(2, 2))
    assert torch.equal(actions[1], torch.full((2, 2), 5.0))


def test_discrete_predict_action_chunk_uses_hf_cached_generation_path():
    class DummyOutput:
        def __init__(self, token_id, batch_size):
            logits = torch.full((batch_size, 1, 128), -1e9)
            logits[:, :, token_id] = 1.0
            self.logits = logits
            self.past_key_values = object()

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.config = SimpleNamespace(
                action_start_token_id=10,
                action_end_token_id=11,
                action_token_start_id=100,
                num_action_tokens=4,
                action_horizon=2,
            )
            self.tokens = [10, 100, 101, 11, 2]
            self.index = 0

        def forward(self, **kwargs):
            batch_size = int(kwargs["input_ids"].shape[0])
            return DummyOutput(self.tokens[self.index], batch_size)

        def _consume_generation_tokens(self, token_ids, *, past_key_values, attention_mask):
            del past_key_values
            self.index += 1
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(token_ids[:, None])], dim=-1)
            return DummyOutput(self.tokens[self.index], int(token_ids.shape[0])), attention_mask

        def _require_eos_token_id(self):
            return 2

        def _action_token_id_to_bin(self):
            return {100: 0, 101: 1, 102: 2, 103: 3}

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        action_mode="discrete",
        inference_action_mode="discrete",
        model_dtype="float32",
        output_features={ACTION: SimpleNamespace(shape=(2,))},
        discrete_generation_max_steps=None,
        discrete_action_tokenizer="unused",
        trust_remote_code=True,
        chunk_size=2,
        n_action_steps=1,
        rtc_config=None,
    )
    policy.model = DummyModel()
    policy.action_tokenizer = _DummyActionTokenizer()

    actions = policy.predict_action_chunk(
        {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
    )

    assert policy.model.index == 4
    assert actions.shape == (1, 1, 2)
    assert torch.equal(actions, torch.ones(1, 1, 2))


def test_discrete_predict_action_chunk_uses_graph_backed_ar_decode_when_enabled():
    class DummyOutput:
        def __init__(self, token_id, past_key_values):
            logits = torch.full((1, 1, 128), -1e9)
            logits[:, :, token_id] = 1.0
            self.logits = logits
            self.past_key_values = past_key_values

    class DummyLmHead(torch.nn.Module):
        def forward(self, hidden_states):
            token_id = int(hidden_states[0, 0, 0].item())
            logits = torch.full((1, 1, 128), -1e9)
            logits[:, :, token_id] = 1.0
            return logits

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.lm_head = DummyLmHead()
            self.config = SimpleNamespace(
                action_start_token_id=10,
                action_end_token_id=11,
                action_token_start_id=100,
                num_action_tokens=4,
                action_horizon=2,
            )
            self.tokens = [10, 100, 101, 11, 2]
            self.index = 0
            self.used_static_cache = False
            self.graph_steps = 0

        def forward(self, **kwargs):
            self.used_static_cache = kwargs.get("past_key_values") == "static-cache"
            return DummyOutput(self.tokens[self.index], kwargs.get("past_key_values"))

        def _make_ar_decode_static_cache(self, inputs, *, max_steps):
            assert int(inputs["input_ids"].shape[1]) == 3
            assert max_steps == 32
            return "static-cache"

        def _make_depth_decode_attention_bias(self, inputs, past_key_values):
            assert past_key_values == "static-cache"
            return torch.ones(1, 1, 35, 35, dtype=torch.float32)

        def _run_ar_decode_step(self, token_ids, *, past_key_values, attention_bias):
            assert past_key_values == "static-cache"
            assert attention_bias.shape == (1, 1, 35, 35)
            self.index += 1
            self.graph_steps += 1
            return torch.tensor([[[float(self.tokens[self.index])]]]), past_key_values

        def _require_eos_token_id(self):
            return 2

        def _action_token_id_to_bin(self):
            return {100: 0, 101: 1, 102: 2, 103: 3}

    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        action_mode="discrete",
        inference_action_mode="discrete",
        model_dtype="float32",
        output_features={ACTION: SimpleNamespace(shape=(2,))},
        discrete_generation_max_steps=None,
        discrete_action_tokenizer="unused",
        trust_remote_code=True,
        chunk_size=2,
        n_action_steps=1,
        rtc_config=None,
        enable_inference_cuda_graph=True,
    )
    policy.model = DummyModel()
    policy.action_tokenizer = _DummyActionTokenizer()
    torch.nn.Module.train(policy, False)

    actions = policy.predict_action_chunk(
        {
            "input_ids": torch.ones(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
    )

    assert policy.model.used_static_cache
    assert policy.model.graph_steps == 4
    assert actions.shape == (1, 1, 2)
    assert torch.equal(actions, torch.ones(1, 1, 2))


class _DummyMolmoBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(5, 3)

    def get_input_embeddings(self):
        return self.embed


class _DummyMolmoModel(torch.nn.Module):
    def __init__(self, *, tie_lm_head: bool = False):
        super().__init__()
        self.model = _DummyMolmoBackbone()
        self.lm_head = torch.nn.Linear(3, 5, bias=False)
        if tie_lm_head:
            self.lm_head.weight = self.model.embed.weight

    def get_input_embeddings(self):
        return self.model.embed


def test_freeze_embedding_freezes_input_embeddings_only_when_untied():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.model = _DummyMolmoModel()

    policy._freeze_input_embeddings()

    assert not policy.model.model.embed.weight.requires_grad
    assert policy.model.lm_head.weight.requires_grad


def test_freeze_embedding_rejects_tied_lm_head_without_mutating():
    policy = object.__new__(MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.model = _DummyMolmoModel(tie_lm_head=True)

    with pytest.raises(RuntimeError, match="would also freeze lm_head"):
        policy._freeze_input_embeddings()

    assert policy.model.model.embed.weight.requires_grad
