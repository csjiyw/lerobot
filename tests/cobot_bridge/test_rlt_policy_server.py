from __future__ import annotations

from dataclasses import asdict
import pytest
import torch

from cobot_bridge.rlt_policy_client import (
    RLT_POLICY_PROTOCOL,
    RLTPolicyServerError,
    decode_rlt_json,
    encode_rlt_json,
    make_rlt_request,
    make_rlt_response,
    validate_rlt_request,
)
from cobot_bridge.rlt_policy_server import (
    LegacyActionConditionedRLTokenEncoder,
    RLTPolicyRuntime,
)
from cobot_bridge.rlt_stage2_data import (
    ACTION_CHUNK_TIMEBASE,
    ACTOR_OUTPUT_SEMANTICS,
    STEAM_OFFLINE_ACTOR_STAGE,
)
from lerobot.policies.smolvla_rlt.configuration_smolvla_rlt import SmolVLARLTConfig
from lerobot.policies.smolvla_rlt.modeling_smolvla_rlt import RLActorMLP


def test_rlt_policy_protocol_round_trip_and_commands():
    request = make_rlt_request(
        "BEGIN_EPISODE",
        workflow_mode=1,
        allow_actor=False,
        actor_exploration_std=0.002,
    )

    decoded = validate_rlt_request(decode_rlt_json(encode_rlt_json(request)))
    response = make_rlt_response(decoded, ok=True, result={"ready": True})

    assert decoded["protocol"] == RLT_POLICY_PROTOCOL
    assert response["request_id"] == request["request_id"]
    assert response["result"]["ready"] is True
    with pytest.raises(RLTPolicyServerError, match="unknown"):
        make_rlt_request("MOVE_ROBOT")


class FakePolicy:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


def _runtime_without_full_model(tmp_path):
    stage1 = (tmp_path / "stage1.pt").resolve()
    stage1.write_bytes(b"stage1 identity")
    runtime = RLTPolicyRuntime.__new__(RLTPolicyRuntime)
    runtime.stage1_path = stage1
    runtime.stage2_path = (tmp_path / "latest.pt").resolve()
    runtime.device = torch.device("cpu")
    runtime.cameras = ("front", "right")
    runtime.base_policy_path = "/base/smolvla"
    runtime.rlt_architecture = "rlinf_prefix_v1"
    runtime.config = SmolVLARLTConfig(
        input_features={},
        output_features={},
        rlt_hidden_dim=8,
        rlt_num_heads=2,
        state_dim=7,
        n_action_steps_rl=3,
        action_dim=7,
        actor_hidden_dims=[8],
        modality_projection_dim=4,
        actor_output_mode="residual",
    )
    runtime.reference_action_steps = int(runtime.config.n_action_steps_rl)
    runtime._requested_action_steps = runtime.reference_action_steps
    runtime.execution_action_steps = runtime.reference_action_steps
    runtime.policy = FakePolicy()
    runtime.actor = RLActorMLP(runtime.config)
    runtime.actor_config = runtime.config
    runtime.actor_output_semantics = ACTOR_OUTPUT_SEMANTICS
    runtime.actor_checkpoint_stage = None
    runtime.actor_read_only = False
    runtime.active_episode = False
    runtime.actor_allowed = False
    runtime.actor_ready = False
    runtime.actor_generation = 0
    runtime.actor_total_updates = 0
    runtime.actor_exploration_std = 0.002
    runtime.actor_action_scale = torch.ones((1, 1, 7))
    runtime.uses_relative_eef_actions = True
    runtime._loaded_stage2_signature = None
    return runtime


def _publish_actor(
    runtime,
    *,
    value: float,
    updates: int,
    ready: bool = True,
    stage: str = "rlt_stage2",
    actor_config: SmolVLARLTConfig | None = None,
):
    config = runtime.config if actor_config is None else actor_config
    actor = RLActorMLP(config)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.fill_(value)
    temporary = runtime.stage2_path.with_name(f"latest-{updates}.tmp")
    torch.save(
        {
            "stage": stage,
            "actor_output_semantics": ACTOR_OUTPUT_SEMANTICS,
            "action_chunk_timebase": ACTION_CHUNK_TIMEBASE,
            "stage1_checkpoint": str(runtime.stage1_path),
            "rlt_config": asdict(config),
            "actor_state_dict": actor.state_dict(),
            "ready_for_online": ready,
            "actor_action_scale": [1.0] * 7,
        },
        temporary,
    )
    payload = torch.load(temporary, map_location="cpu", weights_only=False)
    if stage == "rlt_stage2":
        payload["total_updates"] = updates
    else:
        payload["step"] = updates
    torch.save(payload, temporary)
    temporary.replace(runtime.stage2_path)


def _actor_parameters_equal(actor: torch.nn.Module, value: float) -> bool:
    return all(
        torch.allclose(parameter, torch.full_like(parameter, value))
        for parameter in actor.parameters()
    )


def test_actor_checkpoint_switches_only_at_episode_boundary(tmp_path):
    runtime = _runtime_without_full_model(tmp_path)
    _publish_actor(runtime, value=1.0, updates=10)

    first = runtime.begin_episode(
        workflow_mode=1, allow_actor=True, actor_exploration_std=0.002
    )
    assert first["actor_enabled"] is True
    # A ``latest.pt``-only publication has no numbered historical checkpoint.
    assert first["actor_generation"] == 0
    assert _actor_parameters_equal(runtime.actor, 1.0)

    _publish_actor(runtime, value=2.0, updates=20)
    assert runtime.status()["actor_update_pending"] is True
    assert _actor_parameters_equal(runtime.actor, 1.0)
    with pytest.raises(RLTPolicyServerError, match="already active"):
        runtime.begin_episode(
            workflow_mode=1, allow_actor=True, actor_exploration_std=0.002
        )

    runtime.end_episode()
    second = runtime.begin_episode(
        workflow_mode=2, allow_actor=True, actor_exploration_std=0.002
    )
    assert second["actor_generation"] == 0
    assert second["actor_total_updates"] == 20
    assert second["actor_update_pending"] is False
    assert _actor_parameters_equal(runtime.actor, 2.0)


def test_short_stage2_actor_reuses_long_stage1_reference(tmp_path):
    runtime = _runtime_without_full_model(tmp_path)
    short = SmolVLARLTConfig(
        input_features={},
        output_features={},
        rlt_hidden_dim=8,
        rlt_num_heads=2,
        state_dim=7,
        n_action_steps_rl=2,
        action_dim=7,
        actor_hidden_dims=[8],
        modality_projection_dim=4,
        actor_output_mode="residual",
    )
    _publish_actor(runtime, value=1.0, updates=10, actor_config=short)

    status = runtime.begin_episode(
        workflow_mode=1, allow_actor=True, actor_exploration_std=0.0
    )

    assert status["reference_action_steps"] == 3
    assert status["dimensions"]["action_steps"] == 2
    assert runtime.execution_action_steps == 2

    runtime.end_episode()
    reference_only = runtime.begin_episode(
        workflow_mode=1, allow_actor=False, actor_exploration_std=0.0
    )
    assert reference_only["actor_enabled"] is False
    assert reference_only["dimensions"]["action_steps"] == 3
    assert runtime.execution_action_steps == 3


def test_actor_checkpoint_rejects_legacy_stride_two_timebase(tmp_path):
    runtime = _runtime_without_full_model(tmp_path)
    _publish_actor(runtime, value=1.0, updates=10)
    payload = torch.load(runtime.stage2_path, map_location="cpu", weights_only=False)
    payload["actor_output_semantics"] = "direct_bounded_normalized_action_v2"
    payload.pop("action_chunk_timebase")
    torch.save(payload, runtime.stage2_path)

    with pytest.raises(RuntimeError, match="incompatible actor output semantics"):
        runtime.begin_episode(
            workflow_mode=1, allow_actor=True, actor_exploration_std=0.0
        )


def test_steam_offline_actor_is_deterministic_and_read_only(tmp_path):
    runtime = _runtime_without_full_model(tmp_path)
    _publish_actor(
        runtime,
        value=1.0,
        updates=1,
        stage=STEAM_OFFLINE_ACTOR_STAGE,
    )

    status = runtime.begin_episode(
        workflow_mode=1, allow_actor=True, actor_exploration_std=0.002
    )

    assert status["actor_enabled"] is True
    assert status["actor_read_only"] is True
    assert status["actor_checkpoint_stage"] == STEAM_OFFLINE_ACTOR_STAGE
    assert status["actor_exploration_std"] == 0.0
    assert status["actor_total_updates"] == 1


def test_legacy_action_conditioned_encoder_loads_format_v2_state_strictly():
    source = LegacyActionConditionedRLTokenEncoder(
        vlm_hidden_dim=12,
        expert_hidden_dim=8,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    restored = LegacyActionConditionedRLTokenEncoder(
        vlm_hidden_dim=12,
        expert_hidden_dim=8,
        hidden_dim=16,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
    )
    restored.load_state_dict(source.state_dict(), strict=True)
    source.eval()
    restored.eval()
    vlm = torch.randn(2, 5, 12)
    expert = torch.randn(2, 3, 8)

    assert torch.allclose(source(vlm, expert), restored(vlm, expert))


def test_actor_generation_resets_when_checkpoint_rolls_back(tmp_path):
    runtime = _runtime_without_full_model(tmp_path)
    _publish_actor(runtime, value=1.0, updates=30_000)

    first = runtime.begin_episode(
        workflow_mode=1, allow_actor=True, actor_exploration_std=0.002
    )
    assert first["actor_generation"] == 0
    runtime.end_episode()

    _publish_actor(runtime, value=2.0, updates=30_560)
    forward = runtime.begin_episode(
        workflow_mode=1, allow_actor=True, actor_exploration_std=0.002
    )
    assert forward["actor_generation"] == 0
    runtime.end_episode()

    _publish_actor(runtime, value=1.0, updates=30_000)
    rollback = runtime.begin_episode(
        workflow_mode=1, allow_actor=True, actor_exploration_std=0.002
    )
    assert rollback["actor_total_updates"] == 30_000
    assert rollback["actor_generation"] == 0
    assert _actor_parameters_equal(runtime.actor, 1.0)


def test_actor_rejects_checkpoint_that_failed_ready_for_online_gate(tmp_path):
    runtime = _runtime_without_full_model(tmp_path)
    _publish_actor(runtime, value=1.0, updates=10, ready=False)

    status = runtime.begin_episode(
        workflow_mode=1, allow_actor=True, actor_exploration_std=0.002
    )
    # Operator policy deliberately exposes a held-out-gate failure as a
    # warning, but does not silently disable an explicitly selected actor.
    assert status["actor_enabled"] is True
    assert status["actor_ready"] is True
    assert status["actor_rejection_reason"] is None
    assert runtime.active_episode is True
