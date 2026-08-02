from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from torch.optim.lr_scheduler import LambdaLR

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import (
    AdamWConfig,
    LRSchedulerConfig,
    OptimizerConfig,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from ..rtc.configuration_rtc import RTCConfig

MOLMOACT2_DEFAULT_NUM_IMAGES = 2
MOLMOACT2_IMAGE_TOKENS_PER_IMAGE = 196
MOLMOACT2_FIXED_PROMPT_TOKEN_BUDGET = 80
MOLMOACT2_TASK_TOKEN_BUDGET = 32
MOLMOACT2_SEQUENCE_LENGTH_MARGIN = 32
MOLMOACT2_SEQUENCE_LENGTH_MULTIPLE = 64
MOLMOACT2_DISCRETE_ACTION_WRAPPER_TOKENS = 4
MOLMOACT2_MIN_DISCRETE_ACTION_TOKENS_PER_STEP = 6
MOLMOACT2_DISCRETE_ACTION_TOKENS_PER_DIM = 0.95


@LRSchedulerConfig.register_subclass("molmoact2_cosine_decay_with_warmup")
@dataclass
class MolmoAct2CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Cosine decay with independent warmup for each MolmoAct2 parameter group."""

    peak_lr: float
    decay_lr: float
    num_warmup_steps: int
    num_decay_steps: int | None
    vlm_warmup_steps: int | None = None
    vit_warmup_steps: int | None = None
    connector_warmup_steps: int | None = None
    action_expert_warmup_steps: int | None = None
    goal_warmup_steps: int | None = None
    semantic_visual_warmup_steps: int | None = None

    def build(self, optimizer, num_training_steps: int):
        decay_steps = num_training_steps if self.num_decay_steps is None else self.num_decay_steps
        if decay_steps < 1:
            raise ValueError(f"num_decay_steps must be positive, got {decay_steps}.")

        warmup_by_group = {
            "vlm": self.vlm_warmup_steps,
            "vit": self.vit_warmup_steps,
            "connector": self.connector_warmup_steps,
            "action_expert": self.action_expert_warmup_steps,
            "goal": self.goal_warmup_steps,
            "semantic_visual": self.semantic_visual_warmup_steps,
        }
        lambdas = []
        for group in optimizer.param_groups:
            group_name = str(group.get("name", "vlm"))
            configured_warmup = warmup_by_group.get(group_name)
            warmup_steps = self.num_warmup_steps if configured_warmup is None else configured_warmup
            decay_ratio = (
                min(float(self.decay_lr) / float(self.peak_lr), 1.0) if self.peak_lr > 0 else 1.0
            )

            def lr_lambda(
                current_step: int,
                *,
                warmup_steps: int = int(warmup_steps),
                decay_ratio: float = decay_ratio,
            ) -> float:
                if warmup_steps > 0 and current_step < warmup_steps:
                    return float(current_step + 1) / float(warmup_steps + 1)
                decay_span = max(1, int(decay_steps) - warmup_steps)
                progress = min(max(current_step - warmup_steps, 0) / decay_span, 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return decay_ratio + (1.0 - decay_ratio) * cosine

            lambdas.append(lr_lambda)

        return LambdaLR(optimizer, lr_lambda=lambdas, last_epoch=-1)


def _round_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def infer_molmoact2_max_sequence_length(
    *,
    num_images: int,
    state_dim: int,
    action_dim: int,
    action_horizon: int,
    include_discrete_action: bool,
    num_goal_tokens: int = 0,
) -> int:
    """Infer the padded text/image sequence cap from MolmoAct2's fixed token layout."""
    if num_images < 1:
        num_images = MOLMOACT2_DEFAULT_NUM_IMAGES
    if state_dim < 0:
        state_dim = 0
    if action_dim < 1:
        action_dim = 1
    if action_horizon < 1:
        action_horizon = 1
    if num_goal_tokens < 0:
        num_goal_tokens = 0

    image_tokens = num_images * MOLMOACT2_IMAGE_TOKENS_PER_IMAGE
    prompt_tokens = (
        MOLMOACT2_FIXED_PROMPT_TOKEN_BUDGET
        + MOLMOACT2_TASK_TOKEN_BUDGET
        + state_dim
        + num_goal_tokens
        + MOLMOACT2_SEQUENCE_LENGTH_MARGIN
    )
    action_tokens = 0
    if include_discrete_action:
        action_tokens_per_step = max(
            MOLMOACT2_MIN_DISCRETE_ACTION_TOKENS_PER_STEP,
            math.ceil(action_dim * MOLMOACT2_DISCRETE_ACTION_TOKENS_PER_DIM),
        )
        action_tokens = MOLMOACT2_DISCRETE_ACTION_WRAPPER_TOKENS + action_horizon * action_tokens_per_step

    return _round_up(
        image_tokens + prompt_tokens + action_tokens,
        MOLMOACT2_SEQUENCE_LENGTH_MULTIPLE,
    )


@PreTrainedConfig.register_subclass("molmoact2")
@dataclass
class MolmoAct2Config(PreTrainedConfig):
    """MolmoAct2 policy backed by the converted HF checkpoint implementation."""

    checkpoint_path: str = "allenai/MolmoAct2"
    checkpoint_revision: str | None = None
    checkpoint_force_download: bool = False
    # Optional clean bootstrap: load only VLM tensors from this checkpoint, then
    # reset the action expert initialized by checkpoint_path's MolmoAct2 template.
    vlm_checkpoint_path: str | None = None
    vlm_checkpoint_revision: str | None = None
    vlm_checkpoint_force_download: bool = False
    randomize_action_expert: bool = False
    audit_bootstrap: bool = False
    trust_remote_code: bool = True

    n_obs_steps: int = 1
    chunk_size: int = 30
    n_action_steps: int = 30

    action_mode: str = "both"
    inference_action_mode: str | None = None
    discrete_action_tokenizer: str = "allenai/MolmoAct2-FAST-Tokenizer"
    discrete_generation_max_steps: int | None = None
    norm_tag: str | None = None

    setup_type: str = ""
    control_mode: str = ""
    image_keys: list[str] = field(default_factory=list)
    disable_visual_input: bool = False
    # Goal-pose action prior (two-stage). Goal tokens are K continuous embeddings
    # inserted into the VLM sequence (after language+state); the action expert reads
    # them via its per-layer cross-attention. Stage 1 sources them from an SE(3)
    # encoder over the chunk-end target state; Stage 2 sources them from learnable
    # queries that the VLM contextualizes from vision, supervised by a pose decoder.
    enable_goal_pose: bool = False
    num_goal_tokens: int = 4
    goal_token_source: str = "se3_encoder"  # "se3_encoder" (Stage 1) | "learnable_queries" (Stage 2)
    goal_hidden_dim: int = 512
    # Frame offset (relative to the current frame) whose configured goal feature defines
    # the target pose. Must be set when goal-pose is enabled and a target is needed
    # (Stage 1, or Stage 2 pose reconstruction).
    target_pose_delta_index: int | None = None
    # Dataset feature used for the future goal target. The legacy default keeps existing
    # LIBERO configs/checkpoints unchanged; DROID can point this at an independent 7-D pose.
    goal_pose_feature_key: str = OBS_STATE
    mask_image_from_action_expert: bool = False
    enable_pose_reconstruction: bool = False
    pose_recon_loss_weight: float = 1.0
    init_queries_from_se3_encoder: bool = False
    optimizer_goal_lr: float = 5e-5
    scheduler_goal_warmup_steps: int | None = None
    # v2: a recurrent latent bottleneck shared across all VLM/AE layers. At each
    # layer, the latent queries first cross-attend vision-fused language/state
    # hidden states and then the image patch hidden states. The resulting tokens
    # are appended to the AE context (not to the causal VLM sequence).
    goal_conditioning_mode: str = "vlm_appended"
    num_semantic_visual_tokens: int = 100
    # v2 scheme-2a: the first `num_semantic_visual_pose_tokens` latent tokens form the
    # goal-pose group (supervised by L_pose via a concat readout); the remaining tokens
    # are the context group. All tokens still condition the action expert.
    num_semantic_visual_pose_tokens: int = 8
    semantic_visual_hidden_dim: int = 768
    semantic_visual_num_heads: int = 8
    semantic_visual_ffn_ratio: float = 4.0
    semantic_visual_dropout: float = 0.0
    # Backward-compatible defaults reproduce v2 exactly. v2b enables global latent
    # self-attention and splits the VLM depth into multiple parameter groups.
    semantic_visual_enable_self_attention: bool = False
    semantic_visual_num_layer_groups: int = 1
    optimizer_semantic_visual_lr: float = 1e-5
    scheduler_semantic_visual_warmup_steps: int | None = None
    normalize_language: bool = True
    add_setup_tokens: bool = True
    add_control_tokens: bool = True
    normalize_gripper: bool = False
    num_state_tokens: int = 256
    # Leave unset for the default MolmoAct2 sequence budget inferred from the fixed
    # image/prompt/state/action token layout. Override only for unusual long prompts.
    max_sequence_length: int | None = None

    # Fixed by released MolmoAct2 checkpoints. We validate this at model load.
    expected_max_action_dim: int = 32

    # Flow-matching training knobs copied from the original MolmoAct2 training path.
    num_flow_timesteps: int = 8
    flow_matching_cutoff: float = 1.0
    flow_matching_time_offset: float = 0.001
    flow_matching_time_scale: float = 0.999
    flow_matching_beta_alpha: float = 1.0
    flow_matching_beta_beta: float = 1.5
    num_inference_steps: int | None = None
    mask_action_dim_padding: bool = True
    enable_inference_cuda_graph: bool = True
    # MolmoAct2-local eval option. When enabled, stochastic continuous action
    # generation uses a rollout-local generator derived from eval_seed.
    per_episode_seed: bool = False
    eval_seed: int | None = None
    rtc_config: RTCConfig | None = None

    # Default is full finetuning with gradients from the action expert flowing into the VLM.
    enable_lora_vlm: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    enable_lora_action_expert: bool = False
    enable_knowledge_insulation: bool = False
    freeze_embedding: bool = True
    train_action_expert_only: bool = False
    gradient_checkpointing: bool = False

    model_dtype: str = "bfloat16"
    softmax_auxiliary_loss: bool = True
    softmax_auxiliary_loss_scale: float = 1e-4
    discrete_loss_token_weighting: str = "root_subsegments_root_tokens"

    optimizer_lr: float = 1e-5
    optimizer_vit_lr: float = 5e-6
    optimizer_connector_lr: float = 5e-6
    optimizer_action_expert_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-6
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 200
    scheduler_vlm_warmup_steps: int | None = None
    scheduler_vit_warmup_steps: int | None = None
    scheduler_connector_warmup_steps: int | None = None
    scheduler_action_expert_warmup_steps: int | None = None
    scheduler_decay_steps: int | None = None
    scheduler_decay_lr: float = 1e-6

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)
    dataset_feature_names: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.action_mode not in {"continuous", "discrete", "both"}:
            raise ValueError(
                f"Unsupported action_mode={self.action_mode!r}. "
                "Expected one of {'continuous', 'discrete', 'both'}."
            )
        if self.inference_action_mode not in {None, "continuous", "discrete"}:
            raise ValueError(
                f"Unsupported inference_action_mode={self.inference_action_mode!r}. "
                "Expected one of {None, 'continuous', 'discrete'}."
            )
        if self.inference_action_mode == "continuous" and self.action_mode == "discrete":
            raise ValueError("MolmoAct2 action_mode='discrete' cannot run continuous inference.")
        if self.inference_action_mode == "discrete" and self.action_mode == "continuous":
            raise ValueError("MolmoAct2 action_mode='continuous' cannot run discrete inference.")
        if self.train_action_expert_only and self.action_mode != "continuous":
            raise ValueError("MolmoAct2 train_action_expert_only requires action_mode='continuous'.")
        if self.train_action_expert_only and self.enable_lora_vlm:
            raise ValueError("MolmoAct2 train_action_expert_only is incompatible with enable_lora_vlm.")
        if self.enable_lora_action_expert and not self.enable_lora_vlm:
            raise ValueError("MolmoAct2 enable_lora_action_expert requires enable_lora_vlm.")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}.")
        if self.n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {self.n_action_steps}.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})."
            )
        if self.expected_max_action_dim != 32:
            raise ValueError("MolmoAct2 released checkpoints use expected_max_action_dim=32.")
        if self.model_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError(
                f"Unsupported model_dtype={self.model_dtype!r}. Expected 'float32', 'bfloat16', or 'float16'."
            )
        if self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}.")
        if self.lora_alpha < 1:
            raise ValueError(f"lora_alpha must be >= 1, got {self.lora_alpha}.")
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError(f"lora_dropout must be in [0, 1], got {self.lora_dropout}.")
        if self.lora_bias not in {"none", "all", "lora_only"}:
            raise ValueError(
                f"Unsupported lora_bias={self.lora_bias!r}. Expected one of 'none', 'all', or 'lora_only'."
            )
        if self.discrete_loss_token_weighting not in {
            "none",
            "token",
            "root_tokens",
            "root_subsegments",
            "root_subsegments_root_tokens",
        }:
            raise ValueError(
                f"Unsupported discrete_loss_token_weighting={self.discrete_loss_token_weighting!r}."
            )
        if self.discrete_generation_max_steps is not None and self.discrete_generation_max_steps < 1:
            raise ValueError(
                f"discrete_generation_max_steps must be >= 1 or None, got {self.discrete_generation_max_steps}."
            )
        if self.max_sequence_length is not None and self.max_sequence_length < 1:
            raise ValueError(f"max_sequence_length must be >= 1 or None, got {self.max_sequence_length}.")
        if self.randomize_action_expert and not self.vlm_checkpoint_path:
            raise ValueError("randomize_action_expert=true requires policy.vlm_checkpoint_path.")
        if self.vlm_checkpoint_path and not self.randomize_action_expert:
            raise ValueError(
                "policy.vlm_checkpoint_path requires randomize_action_expert=true so released "
                "MolmoAct2 action-expert weights cannot leak into a clean bootstrap."
            )
        for name, value in (
            ("scheduler_warmup_steps", self.scheduler_warmup_steps),
            ("scheduler_vlm_warmup_steps", self.scheduler_vlm_warmup_steps),
            ("scheduler_vit_warmup_steps", self.scheduler_vit_warmup_steps),
            ("scheduler_connector_warmup_steps", self.scheduler_connector_warmup_steps),
            ("scheduler_action_expert_warmup_steps", self.scheduler_action_expert_warmup_steps),
            ("scheduler_goal_warmup_steps", self.scheduler_goal_warmup_steps),
            (
                "scheduler_semantic_visual_warmup_steps",
                self.scheduler_semantic_visual_warmup_steps,
            ),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if self.goal_token_source not in {"se3_encoder", "learnable_queries"}:
            raise ValueError(
                f"Unsupported goal_token_source={self.goal_token_source!r}. "
                "Expected 'se3_encoder' or 'learnable_queries'."
            )
        if not str(self.goal_pose_feature_key).strip():
            raise ValueError("goal_pose_feature_key must be a non-empty dataset feature key.")
        if self.goal_conditioning_mode not in {"vlm_appended", "semantic_visual_recurrent"}:
            raise ValueError(
                f"Unsupported goal_conditioning_mode={self.goal_conditioning_mode!r}. "
                "Expected 'vlm_appended' or 'semantic_visual_recurrent'."
            )
        if self.num_semantic_visual_tokens < 1:
            raise ValueError(
                "num_semantic_visual_tokens must be >= 1, "
                f"got {self.num_semantic_visual_tokens}."
            )
        if self.semantic_visual_hidden_dim < 1:
            raise ValueError(
                f"semantic_visual_hidden_dim must be >= 1, got {self.semantic_visual_hidden_dim}."
            )
        if self.semantic_visual_num_heads < 1:
            raise ValueError(
                f"semantic_visual_num_heads must be >= 1, got {self.semantic_visual_num_heads}."
            )
        if self.semantic_visual_hidden_dim % self.semantic_visual_num_heads != 0:
            raise ValueError(
                "semantic_visual_hidden_dim must be divisible by semantic_visual_num_heads, "
                f"got {self.semantic_visual_hidden_dim} and {self.semantic_visual_num_heads}."
            )
        if self.semantic_visual_ffn_ratio <= 0:
            raise ValueError(
                f"semantic_visual_ffn_ratio must be > 0, got {self.semantic_visual_ffn_ratio}."
            )
        if not 0 <= self.semantic_visual_dropout <= 1:
            raise ValueError(
                f"semantic_visual_dropout must be in [0, 1], got {self.semantic_visual_dropout}."
            )
        if self.semantic_visual_num_layer_groups < 1:
            raise ValueError(
                "semantic_visual_num_layer_groups must be >= 1, "
                f"got {self.semantic_visual_num_layer_groups}."
            )
        if self.enable_goal_pose:
            if self.num_goal_tokens < 1:
                raise ValueError(f"num_goal_tokens must be >= 1, got {self.num_goal_tokens}.")
            if self.pose_recon_loss_weight < 0:
                raise ValueError(
                    f"pose_recon_loss_weight must be non-negative, got {self.pose_recon_loss_weight}."
                )
            if self.action_mode != "continuous":
                raise ValueError("MolmoAct2 enable_goal_pose requires action_mode='continuous'.")
            needs_target = self.goal_token_source == "se3_encoder" or self.enable_pose_reconstruction
            if needs_target and (self.target_pose_delta_index is None or self.target_pose_delta_index < 1):
                raise ValueError(
                    "enable_goal_pose with goal_token_source='se3_encoder' or "
                    "enable_pose_reconstruction=true requires target_pose_delta_index >= 1 "
                    "(e.g. chunk_size for s_{t+H})."
                )
            if self.goal_token_source == "se3_encoder" and self.enable_pose_reconstruction:
                raise ValueError(
                    "goal_token_source='se3_encoder' (Stage 1) does not use pose reconstruction; "
                    "set enable_pose_reconstruction=false."
                )
            if self.goal_conditioning_mode == "semantic_visual_recurrent":
                if self.goal_token_source != "learnable_queries":
                    raise ValueError(
                        "semantic_visual_recurrent requires goal_token_source='learnable_queries'."
                    )
                if self.disable_visual_input:
                    raise ValueError("semantic_visual_recurrent requires visual input.")
                if not self.mask_image_from_action_expert:
                    raise ValueError(
                        "semantic_visual_recurrent requires mask_image_from_action_expert=true; "
                        "raw image KV must reach the AE only through the latent aggregator."
                    )
                if not 1 <= self.num_semantic_visual_pose_tokens < self.num_semantic_visual_tokens:
                    raise ValueError(
                        "num_semantic_visual_pose_tokens must satisfy "
                        "1 <= num_semantic_visual_pose_tokens < num_semantic_visual_tokens, got "
                        f"{self.num_semantic_visual_pose_tokens} vs {self.num_semantic_visual_tokens}."
                    )
        elif (
            self.mask_image_from_action_expert
            or self.enable_pose_reconstruction
            or self.goal_conditioning_mode == "semantic_visual_recurrent"
        ):
            raise ValueError(
                "mask_image_from_action_expert / enable_pose_reconstruction / "
                "semantic_visual_recurrent require enable_goal_pose=true."
            )

    def inferred_max_sequence_length(
        self,
        *,
        num_images: int | None = None,
        state_dim: int | None = None,
        action_dim: int | None = None,
        action_horizon: int | None = None,
        include_discrete_action: bool | None = None,
    ) -> int:
        if self.max_sequence_length is not None:
            return int(self.max_sequence_length)

        if num_images is None:
            num_images = len(self.image_keys) or len(self.image_features) or MOLMOACT2_DEFAULT_NUM_IMAGES
        if state_dim is None:
            state_feature = self.robot_state_feature
            state_dim = int(state_feature.shape[0]) if state_feature is not None else 0
        if action_dim is None:
            action_feature = self.action_feature
            action_dim = (
                int(action_feature.shape[0]) if action_feature is not None else self.expected_max_action_dim
            )
        if action_horizon is None:
            action_horizon = self.chunk_size
        if include_discrete_action is None:
            include_discrete_action = self.action_mode in {"discrete", "both"}

        return infer_molmoact2_max_sequence_length(
            num_images=int(num_images),
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            action_horizon=int(action_horizon),
            include_discrete_action=bool(include_discrete_action),
            num_goal_tokens=(
                int(self.num_goal_tokens)
                if self.enable_goal_pose and self.goal_conditioning_mode == "vlm_appended"
                else 0
            ),
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return MolmoAct2CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            vlm_warmup_steps=self.scheduler_vlm_warmup_steps,
            vit_warmup_steps=self.scheduler_vit_warmup_steps,
            connector_warmup_steps=self.scheduler_connector_warmup_steps,
            action_expert_warmup_steps=self.scheduler_action_expert_warmup_steps,
            goal_warmup_steps=self.scheduler_goal_warmup_steps,
            semantic_visual_warmup_steps=self.scheduler_semantic_visual_warmup_steps,
        )

    def set_dataset_feature_metadata(self, features: dict[str, Any]) -> None:
        self.dataset_feature_names = {}
        for key in dict.fromkeys((ACTION, OBS_STATE, self.goal_pose_feature_key)):
            feature = features.get(key) if isinstance(features, dict) else None
            if isinstance(feature, dict) and feature.get("names") is not None:
                self.dataset_feature_names[key] = feature["names"]

    def validate_features(self) -> None:
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(0,))
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(0,))
