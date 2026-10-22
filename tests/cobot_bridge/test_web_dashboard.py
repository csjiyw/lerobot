from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from fastapi.testclient import TestClient

import cobot_bridge.web_dashboard as web_dashboard
from cobot_bridge.web_dashboard import (
    DashboardConfig,
    GROOT_EEF_POLICY_ID,
    JobManager,
    ManagedJob,
    _device_status,
    _policy_files,
    _raw_files,
    create_app,
)


class FakeManager:
    def __init__(self):
        self.started = []
        self.jobs = []
        self.motion_stopped = False

    def list(self):
        return self.jobs

    def start(self, kind, command, metadata=None):
        self.started.append((kind, command))
        job = {
            "id": "test-job",
            "kind": kind,
            "state": "running",
            "metadata": dict(metadata or {}),
        }
        self.jobs = [job]
        return job

    def stop(self, job_id):
        return {"id": job_id, "state": "stopping"}

    def log(self, _job_id):
        return "test log"

    def stop_motion_jobs(self):
        self.motion_stopped = True


class FakeDeviceMonitor:
    def probe(self):
        return {
            "online": True,
            "allow_control": True,
            "teach_active": False,
            "emergency_latched": False,
            "devices": {"ok": True, "ok_count": 7, "total": 7},
        }

    def image(self, name):
        return f"jpeg-{name}".encode(), "image/jpeg"


def test_default_checkpoint_root_is_training_outputs(monkeypatch):
    monkeypatch.delenv("COBOT_CHECKPOINT_ROOT", raising=False)
    monkeypatch.delenv("COBOT_LEROBOT_REPLAY_ROOT", raising=False)

    config = DashboardConfig.from_env()

    assert config.checkpoint_root == config.workspace / "outputs"
    assert config.lerobot_replay_root == Path.home() / "datasets" / "lerobot_v21"


def test_rlt_page_restores_resident_server_actor_after_refresh(tmp_path):
    client, _manager = make_client(tmp_path)

    page = client.get("/")
    script = client.get("/static/app.js")

    assert page.status_code == 200
    assert 'app.js?v=' in page.text
    assert 'id="replayDataset"' in page.text
    assert 'id="stopReplayButton" class="button danger" disabled>Pause / SAFE</button>' in page.text
    assert '<option value="0.0001">0.0001 · ~0.01–0.014 mm / 0.001°</option>' in page.text
    assert script.status_code == 200
    assert 'serverActorCheckpointId = rltServer.checkpoint_id === selected?.id' in script.text
    assert 'select.value = serverActorCheckpointId' in script.text
    assert 'Recovered · Actor published' in script.text


def test_runtime_notifications_keep_manual_refresh_and_validate_cursor(tmp_path):
    client, _manager = make_client(tmp_path)
    page = client.get('/')
    script = client.get('/static/app.js')
    assert 'id="runtimeEventHint"' in page.text
    assert 'fetch("/api/events"' in script.text
    assert 'Last-Event-ID' in script.text
    assert 'setInterval(' not in script.text
    assert client.get('/api/events', headers={'Last-Event-ID':'invalid'}).status_code == 400


def test_sync_status_is_independent_left_sidebar(tmp_path):
    client, _ = make_client(tmp_path)
    page = client.get('/').text
    css = client.get('/static/styles.css').text
    start = page.index('<aside class="control-sidebar"')
    end = page.index('</aside>', start)
    sidebar = page[start:end]
    assert end < page.index('<main class="dashboard-shell">')
    for name in ('moveRltInitPoseButton', 'safeButton', 'bridgeBadge', 'modeValue', 'robotStatusHint'):
        assert f'id="{name}"' in sidebar
    assert page.count('id="robotStatusHint"') == 1
    assert 'grid-template-columns: 250px minmax(0, 1fr)' in css
    assert '.dashboard-workspace > main { width: 100%; min-width: 0;' in css
    assert '.control-sidebar { position: static;' in css


def test_priority_status_probe_excludes_joints_images_and_scans(tmp_path, monkeypatch):
    client, _ = make_client(tmp_path)
    config = client.app.state.dashboard_config
    calls = []
    robot = {'control': {'effective_mode':'SAFE', 'effective_mode_id':0, 'teach_active':False,
                         'output_enabled':False, 'emergency_latched':False, 'epoch':1}, 'ready':True,
             'master':{}, 'puppet':{}, 'images':{}}
    for role in ('master', 'puppet'):
        robot[role] = {s: {'fresh':True, 'position':[0.]*7, 'age_ms':1.} for s in ('left','right')}
    robot['images'] = {n: {'fresh':True, 'stamp_ns':123} for n in ('front','left','right')}
    class StatusClient:
        def __init__(self, endpoint, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def ping(self): return {'allow_control':True}
        def get_observation(self, **kw):
            calls.append(kw)
            return robot, {}
    monkeypatch.setattr(web_dashboard, 'CobotBridgeClient', StatusClient)
    monitor = web_dashboard.DeviceMonitor(config)
    first = monitor.control_status()
    robot['puppet']['right']['position'][0] = 1.
    robot['puppet']['right']['age_ms'] = 3.
    assert monitor.control_status() == first
    before = len(calls)
    joints = monitor.joint_state()
    assert joints['arms']['puppet_right']['position'][0] == 1.
    assert joints['revision'] == 2
    monitor.joint_state()
    assert len(calls) == before  # Local cache reads must not query the robot.
    assert calls[0]['images'] == ()
    assert 'position' not in json.dumps(first)
    assert 'stamp_ns' not in json.dumps(first)
    assert first['devices']['ok_count'] == 7
    robot['control']['teach_active'] = True
    assert monitor.control_status()['teach_active'] is True
    robot['control'] = {}
    assert monitor.control_status()['allow_control'] is False


def test_priority_sse_frontend_expiry_and_no_directory_refresh(tmp_path):
    client, _ = make_client(tmp_path)
    script = client.get('/static/app.js').text
    assert 'id="robotStatusHint"' in client.get('/').text
    assert 'robotStatusStale("SSE 连接断开")' in script
    assert 'else mergeRobotStatus(overview)' in script
    assert 'robot_status_changed' in script and 'robot_heartbeat' in script
    safety = script.split('async function emergencySafe()')[1].split('async function restartDevices()')[0]
    assert 'refresh()' not in safety


def test_joint_cache_age_and_api_auth(tmp_path, monkeypatch):
    client, manager = make_client(tmp_path)
    config = replace(client.app.state.dashboard_config, api_token='secret')
    monitor = web_dashboard.DeviceMonitor(config)
    now = [10.]
    monkeypatch.setattr(web_dashboard.time, 'monotonic', lambda: now[0])
    assert monitor.joint_state()['revision'] == 0
    monitor._joint_cached_at = 10.
    monitor._joint_cache = {'revision':1, 'arms': {'puppet_right': {'position':[0.]*7, 'fresh':True, 'age_ms':2.}}}
    assert monitor.joint_state()['arms']['puppet_right']['fresh'] is True
    now[0] = 12.
    aged = monitor.joint_state()
    assert aged['age_ms'] == 2000
    assert aged['arms']['puppet_right']['fresh'] is False
    assert aged['arms']['puppet_right']['age_ms'] == 2002
    local = TestClient(create_app(config, manager, monitor))
    assert local.get('/api/robot/joints').status_code == 401
    assert local.get('/api/robot/joints', headers={'X-Cobot-Token':'secret'}).json()['revision'] == 1


def test_joint_feedback_has_own_low_rate_handler(tmp_path):
    client, _ = make_client(tmp_path)
    script = client.get('/static/app.js').text
    assert 'if (kind === "joint_state") { handleJointState(data); return; }' in script
    assert '低频实时' in script
    assert 'setTimeout(jointStateStale, 2000)' in script
    body = script.split('function handleJointState(data)')[1].split('function renderCapabilities')[0]
    assert 'refresh()' not in body


def make_client(tmp_path: Path):
    workspace = Path(__file__).resolve().parents[2]
    raw_root = tmp_path / "raw"
    dataset_root = tmp_path / "datasets"
    lerobot_replay_root = tmp_path / "lerobot_v21"
    checkpoint_root = tmp_path / "checkpoints"
    for root in (raw_root, dataset_root, lerobot_replay_root, checkpoint_root):
        root.mkdir()
    raw = raw_root / "episode.raw.h5"
    raw.write_bytes(b"placeholder")
    dataset = dataset_root / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(
        json.dumps({"total_frames": 1, "total_episodes": 1, "fps": 30})
    )
    replay_dataset = (
        lerobot_replay_root
        / "cobotmagic"
        / "5482b62ee3ff498a92f139f0788c0544_head_eef_joint"
    )
    (replay_dataset / "meta").mkdir(parents=True)
    (replay_dataset / "meta/info.json").write_text(
        json.dumps(
            {
                "total_frames": 5,
                "total_episodes": 1,
                "fps": 30,
                "robot_type": "cobotmagic",
                "features": {
                    "action_joint": {
                        "names": {
                            "motors": [
                                "l-j1", "l-j2", "l-j3", "l-j4", "l-j5", "l-j6",
                                "r-j1", "r-j2", "r-j3", "r-j4", "r-j5", "r-j6",
                                "left_gripper", "right_gripper",
                            ]
                        }
                    }
                },
            }
        )
    )
    policy = checkpoint_root / "policy"
    policy.mkdir()
    (policy / "config.json").write_text(json.dumps({"type": "smolvla"}))
    (policy / "model.safetensors").write_bytes(b"")
    config = DashboardConfig(
        workspace=workspace,
        raw_root=raw_root,
        dataset_root=dataset_root,
        checkpoint_root=checkpoint_root,
        run_root=tmp_path / "runs",
        python=Path("/test/python"),
        lerobot_replay_root=lerobot_replay_root,
        debug_root=tmp_path / "debug_rollouts",
        debug_capture_enabled=False,
        rlt_root=tmp_path / "rlt",
        endpoint="tcp://127.0.0.1:1",
        policy_server_endpoint="tcp://127.0.0.1:1",
        rlt_policy_server_endpoint="tcp://127.0.0.1:2",
    )
    manager = FakeManager()
    return TestClient(create_app(config, manager)), manager


def test_device_status_requires_fresh_7d_arms_and_fresh_camera_frames():
    joint = {"fresh": True, "age_ms": 12.0, "position": list(range(7))}
    image = {"fresh": True, "age_ms": 8.0, "stamp_ns": 123, "encoding": "jpeg"}
    raw = {
        "sequence": 42,
        "master": {"left": joint, "right": joint},
        "puppet": {"left": joint, "right": joint},
        "images": {"front": image, "left": image, "right": image},
    }
    frames = {"front": b"f", "left": b"l", "right": b"r"}

    status = _device_status(raw, frames)

    assert status["ok"] is True
    assert status["ok_count"] == 7
    assert all(arm["ok"] for arm in status["arms"].values())
    assert all(camera["ok"] for camera in status["cameras"].values())

    raw["puppet"]["right"] = {**joint, "fresh": False}
    degraded = _device_status(raw, frames)
    assert degraded["ok"] is False
    assert degraded["ok_count"] == 6
    assert degraded["arms"]["puppet_right"]["ok"] is False


def test_camera_endpoint_serves_cached_frame(tmp_path):
    client, _ = make_client(tmp_path)
    client.app.state.device_monitor = FakeDeviceMonitor()
    # Route closures retain the monitor passed at app creation, so build one explicitly.
    config = client.app.state.dashboard_config
    monitor = FakeDeviceMonitor()
    client = TestClient(create_app(config, FakeManager(), monitor))

    response = client.get("/api/cameras/front")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"jpeg-front"


def test_dashboard_exposes_rollout_speed_control(tmp_path):
    client, _ = make_client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="rolloutFps"' in response.text
    assert '<option value="30" selected>1.0× · 30 FPS</option>' in response.text


def test_dashboard_and_fixed_record_command(tmp_path):
    client, manager = make_client(tmp_path)
    page = client.get("/")
    assert page.status_code == 200
    for tab in ("il", "rl", "rlt", "data", "jobs", "system"):
        assert f'data-tab="{tab}"' in page.text
        assert f'data-panel="{tab}"' in page.text
    assert 'data-tab="replay"' not in page.text
    assert 'data-panel="replay"' not in page.text
    assert 'data-tab="rollout"' not in page.text
    assert 'id="saveRlSuccessButton"' in page.text
    assert 'id="saveRlFailureButton"' in page.text
    assert 'id="abortRlButton"' in page.text
    assert 'id="ilTeachAutosave"' in page.text
    assert 'id="saveIlSuccessButton"' in page.text
    assert 'id="saveIlFailureButton"' in page.text
    assert 'id="abortIlButton"' in page.text
    assert 'id="rlTeachAutosave"' in page.text
    assert 'id="rlAutoStartPolicy"' in page.text
    assert 'id="rlPolicyMode"' in page.text
    assert 'id="rlCollectionWorkflow"' not in page.text
    assert '<option value="1">Mode 1 · right puppet only</option>' in page.text
    assert '<option value="2" selected>Mode 2 · right master + puppet</option>' in page.text
    assert 'id="moveInitPoseButton"' in page.text
    assert 'id="startIlWithInitButton"' in page.text
    assert 'id="moveRlInitPoseButton"' in page.text
    assert 'id="startPolicyServerButton"' in page.text
    assert 'id="stopPolicyServerButton"' in page.text
    assert 'id="policyDenoisingSteps"' in page.text
    assert 'value="5">5 steps · faster' in page.text
    assert 'value="10" selected>10 steps · checkpoint default' in page.text
    assert '<details class="rlt-checks">' in page.text
    assert "Checks &amp; Diagnostics" in page.text
    assert 'id="startRltServerButton"' in page.text
    assert 'id="startRltWarmupButton"' not in page.text
    assert 'id="rltWarmupBadge"' not in page.text
    assert 'id="startRltMode1Button"' in page.text
    assert 'id="startRolloutButton"' in page.text
    assert 'id="startSyncRolloutButton"' in page.text
    assert 'id="stopRolloutButton"' in page.text
    assert 'id="rltCheckpointSelect"' in page.text
    assert 'id="rltActorCheckpointSelect"' in page.text
    assert 'id="validateRltButton"' in page.text
    assert 'id="startRltShadowButton"' in page.text
    assert 'id="stopRltShadowButton"' in page.text
    assert 'id="moveRltInitPoseButton"' in page.text
    assert page.text.count('id="moveRltInitPoseButton"') == 1
    assert page.text.index('id="moveRltInitPoseButton"') < page.text.index('id="safeButton"')
    assert 'id="startRltMode1Button"' in page.text
    assert 'id="startRltMode2Button"' in page.text
    assert 'id="rltPipelineBadge"' in page.text
    assert 'id="rltMotionConfirm"' in page.text
    assert 'id="rltOnlineCollection"' in page.text
    assert "Pure Online RL" in page.text
    assert "Human Intervention" in page.text
    assert 'id="saveRltSuccessButton"' in page.text
    assert 'id="saveRltFailureButton"' in page.text
    assert 'id="abortRltButton"' in page.text
    assert "Success · Reward 1" in page.text
    assert "Failure · Reward 0" in page.text
    assert "Auto-start selected policy mode with recorder" in page.text
    assert 'id="exitStopButton"' in page.text
    assert 'id="restartDevicesButton"' in page.text
    assert 'id="policyRunSelect"' in page.text
    assert 'id="convertRawCollection"' in page.text
    assert 'id="convertActionSource"' in page.text
    assert 'id="convertTrimHolds"' in page.text
    assert 'id="rawEpisodeList"' in page.text
    assert 'id="episodeDetailTask"' in page.text
    assert 'id="episodeDetailOutcome"' in page.text
    assert 'id="startReplayButton"' in page.text
    assert 'id="stopReplayButton"' in page.text
    assert 'id="replayRaw"' not in page.text
    assert 'id="policyDataset"' not in page.text
    assert "Normalization dataset" not in page.text

    response = client.post(
        "/api/jobs/record",
        json={
            "collection": "il_test",
            "task": "pick object",
            "side": "right",
            "cameras": ["front", "right"],
            "fps": 30,
            "stop_on_teach_release": True,
            "collection_type": "IL",
        },
    )

    assert response.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "record"
    assert command[:3] == ["/test/python", "-m", "cobot_bridge.record_raw"]
    assert "--stop-on-teach-release" in command
    assert command[command.index("--collection-type") + 1] == "IL"
    assert "pick object" in command


def test_policy_inventory_groups_checkpoints_by_training_run(tmp_path):
    client, _ = make_client(tmp_path)
    config = client.app.state.dashboard_config
    for step in ("010000", "150000"):
        root = config.checkpoint_root / "training_run" / "checkpoints" / step / "pretrained_model"
        root.mkdir(parents=True)
        (root / "config.json").write_text(json.dumps({"type": "smolvla"}))
        (root / "model.safetensors").write_bytes(b"")

    policies = _policy_files(config)
    grouped = [item for item in policies if item["run_id"] == "training_run"]

    assert {item["step"] for item in grouped} == {10000, 150000}
    assert {item["checkpoint"] for item in grouped} == {"010000", "150000"}
    assert all(item["type"] == "smolvla" for item in grouped)


def test_policy_inventory_discovers_sharded_groot_model(tmp_path):
    client, _ = make_client(tmp_path)
    config = client.app.state.dashboard_config
    root = config.checkpoint_root / "groot_run"
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"architectures": ["Gr00tN1d6"]}))
    (root / "model.safetensors.index.json").write_text("{}")

    policies = _policy_files(config)

    entry = next(item for item in policies if item["id"] == "groot_run")
    assert entry["type"] == "groot"


def test_registered_external_groot_eef_checkpoint_is_selectable_and_launchable(
    tmp_path, monkeypatch
):
    client, manager = make_client(tmp_path)
    config = client.app.state.dashboard_config
    external = tmp_path / "outside_outputs" / "checkpoint-90000"
    external.mkdir(parents=True)
    (external / "config.json").write_text(
        json.dumps({"architectures": ["Gr00tN1d7"]})
    )
    (external / "model.safetensors.index.json").write_text("{}")
    (external / "processor_config.json").write_text(
        json.dumps({"processor_kwargs": {"modality_configs": {"cobotmagic": {}}}})
    )
    backbone = tmp_path / "cosmos_backbone"
    backbone.mkdir()
    config = replace(
        config,
        groot_eef_checkpoint_path=external,
        groot_backbone_path=backbone,
    )
    client = TestClient(create_app(config, manager))
    monkeypatch.setattr(
        web_dashboard,
        "_policy_server_status",
        lambda *_args: {
            "ready": False,
            "state": "offline",
            "endpoint": "tcp://127.0.0.1:1",
        },
    )

    inventory = _policy_files(config)
    registered = next(item for item in inventory if item["id"] == GROOT_EEF_POLICY_ID)
    assert registered["type"] == "groot"
    assert registered["step"] == 90000

    response = client.post(
        "/api/policy-server/start",
        json={"policy_id": GROOT_EEF_POLICY_ID, "device": "cuda:1", "compute": "pc"},
    )

    assert response.status_code == 200
    _, command = manager.started[-1]
    assert command[command.index("--model-path") + 1] == str(external)
    assert command[command.index("--backbone-path") + 1] == str(backbone)


def test_raw_inventory_exposes_collection_and_episode(tmp_path):
    client, _ = make_client(tmp_path)
    config = client.app.state.dashboard_config
    episode = config.raw_root / "collection_a" / "episode_000123.raw.h5"
    episode.parent.mkdir()
    episode.write_bytes(b"not-hdf5")

    entry = next(item for item in _raw_files(config) if item["id"].startswith("collection_a/"))

    assert entry["collection"] == "collection_a"
    assert entry["episode"] == "episode_000123.raw.h5"
    assert entry["modified_at_ns"] > 0


def test_rl_failure_button_finalizes_recorder_and_stops_motion(tmp_path):
    client, manager = make_client(tmp_path)
    started = client.post(
        "/api/jobs/record",
        json={
            "collection": "rl_test",
            "task": "pick object",
            "collection_type": "RL",
            "stop_on_teach_release": False,
        },
    )
    assert started.status_code == 200
    _, command = manager.started[-1]
    assert "--stop-on-teach-release" not in command
    result_file = Path(manager.jobs[0]["metadata"]["result_file"])

    saved = client.post(
        "/api/jobs/test-job/finalize-record",
        json={"outcome": "failure"},
    )

    assert saved.status_code == 200
    assert saved.json()["outcome"] == "failure"
    assert result_file.read_text() == "failure"
    assert manager.motion_stopped is True


def test_il_manual_finish_can_save_failure_without_stopping_motion(tmp_path):
    client, manager = make_client(tmp_path)
    started = client.post(
        "/api/jobs/record",
        json={
            "collection": "il_manual",
            "task": "pick object",
            "collection_type": "IL",
            "stop_on_teach_release": False,
        },
    )
    assert started.status_code == 200
    result_file = Path(manager.jobs[0]["metadata"]["result_file"])

    saved = client.post(
        "/api/jobs/test-job/finalize-record",
        json={"outcome": "failure"},
    )

    assert saved.status_code == 200
    assert saved.json()["collection_type"] == "IL"
    assert saved.json()["outcome"] == "failure"
    assert result_file.read_text() == "failure"
    assert manager.motion_stopped is False


def test_rl_abort_requests_discard_and_stops_motion(tmp_path):
    client, manager = make_client(tmp_path)
    started = client.post(
        "/api/jobs/record",
        json={
            "collection": "rl_abort",
            "task": "pick object",
            "collection_type": "RL",
            "stop_on_teach_release": False,
        },
    )
    assert started.status_code == 200
    result_file = Path(manager.jobs[0]["metadata"]["result_file"])

    aborted = client.post("/api/jobs/test-job/abort-record")

    assert aborted.status_code == 200
    assert aborted.json()["discarding"] is True
    assert result_file.read_text() == "discard"
    assert manager.motion_stopped is True


def test_il_abort_requests_discard_without_stopping_motion(tmp_path):
    client, manager = make_client(tmp_path)
    started = client.post(
        "/api/jobs/record",
        json={
            "collection": "il_abort",
            "task": "pick object",
            "collection_type": "IL",
            "stop_on_teach_release": False,
        },
    )
    assert started.status_code == 200
    result_file = Path(manager.jobs[0]["metadata"]["result_file"])

    aborted = client.post("/api/jobs/test-job/abort-record")

    assert aborted.status_code == 200
    assert result_file.read_text() == "discard"
    assert manager.motion_stopped is False


def test_real_motion_requires_confirmation(tmp_path):
    client, manager = make_client(tmp_path)

    denied = client.post(
        "/api/jobs/replay",
        json={"raw_id": "episode.raw.h5", "execute": True, "confirm_motion": False},
    )
    assert denied.status_code == 400
    assert manager.started == []

    allowed = client.post(
        "/api/jobs/replay",
        json={"raw_id": "episode.raw.h5", "execute": True, "confirm_motion": True},
    )
    assert allowed.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "replay_execute"
    assert "--execute" in command
    assert command[command.index("--source") + 1] == "puppet_command"
    assert command[command.index("--waypoint-hz") + 1] == "30"


def test_lerobot_replay_is_right_arm_only_and_requires_confirmation(tmp_path, monkeypatch):
    client, manager = make_client(tmp_path)
    dataset_id = "cobotmagic/5482b62ee3ff498a92f139f0788c0544_head_eef_joint"
    replay_h5 = tmp_path / "converted.raw.h5"
    monkeypatch.setattr(
        web_dashboard,
        "export_episode_to_raw",
        lambda *_args, **_kwargs: {
            "path": str(replay_h5),
            "frames": 123,
            "fps": 30,
            "task": "insert workpiece",
            "side": "right",
            "source_feature": "action_joint",
            "joint_dimensions": ["r-j1", "r-j2", "r-j3", "r-j4", "r-j5", "r-j6", "right_gripper"],
        },
    )

    denied = client.post(
        "/api/jobs/replay-lerobot",
        json={"dataset_id": dataset_id, "episode_index": 0, "execute": True},
    )
    assert denied.status_code == 400
    assert manager.started == []

    overview = client.get("/api/overview")
    assert overview.status_code == 200
    assert overview.json()["lerobot_replay_datasets"][0]["id"] == dataset_id

    allowed = client.post(
        "/api/jobs/replay-lerobot",
        json={
            "dataset_id": dataset_id,
            "episode_index": 0,
            "side": "right",
            "execute": True,
            "confirm_motion": True,
        },
    )
    assert allowed.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "replay_execute"
    assert command[command.index("--side") + 1] == "right"
    assert command[command.index("--mode") + 1] == "puppet"
    assert command[command.index("--source") + 1] == "puppet_command"
    assert "--no-trim-stationary-holds" in command
    assert command[command.index("--alignment-speed") + 1] == "0.05"
    assert command[command.index("--alignment-min-seconds") + 1] == "5.0"
    assert command[-1] == "--execute"
    assert allowed.json()["metadata"]["dataset_id"] == dataset_id
    assert allowed.json()["metadata"]["episode_index"] == 0


def test_stop_motion_job_requests_safe_output_before_signalling(tmp_path, monkeypatch):
    client, _fake_manager = make_client(tmp_path)
    config = client.app.state.dashboard_config
    manager = JobManager(config)

    class RunningProcess:
        pid = 67890

        @staticmethod
        def poll():
            return None

    class NoStartThread:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            pass

    job = ManagedJob(
        id="replay-job",
        kind="replay_execute",
        command=[],
        log_path=tmp_path / "replay.log",
        started_at_ns=1,
        process=RunningProcess(),
        metadata={},
    )
    manager._jobs[job.id] = job
    safety_calls = []
    signals = []
    monkeypatch.setattr(
        web_dashboard,
        "request_robot_safe_stop",
        lambda endpoint, **_kwargs: safety_calls.append(endpoint) or {
            "output_disabled": {},
            "mode_safe": {},
            "safe": True,
        },
    )
    monkeypatch.setattr(web_dashboard.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(web_dashboard.threading, "Thread", NoStartThread)

    result = manager.stop(job.id)

    assert safety_calls == [config.endpoint]
    assert signals and signals[0][0] == job.process.pid
    assert result["metadata"]["stop_safety"]["safe"] is True


def test_restart_devices_starts_managed_recovery_job(tmp_path):
    client, manager = make_client(tmp_path)
    client = TestClient(
        create_app(client.app.state.dashboard_config, manager, FakeDeviceMonitor())
    )

    response = client.post("/api/system/restart-devices")

    assert response.status_code == 200
    assert manager.motion_stopped is True
    kind, command = manager.started[-1]
    assert kind == "device_restart"
    assert command[:3] == ["/test/python", "-m", "cobot_bridge.restart_devices"]
    assert "--ssh-host" in command
    assert "--remote-workspace" in command


def test_robot_startup_available_when_bridge_offline(tmp_path):
    client, manager = make_client(tmp_path)
    response = client.post('/api/system/start-robot', json={})
    assert response.status_code == 200
    kind, command = manager.started[-1]
    assert kind == 'robot_startup'
    assert command[:3] == ['/test/python', '-m', 'cobot_bridge.ensure_robot']
    assert '--report' in command
    assert manager.motion_stopped is False


def test_robot_self_check_command_and_ui(tmp_path):
    client, manager = make_client(tmp_path)
    response = client.post('/api/system/start-robot', json={'check_only': True})
    assert response.status_code == 200
    assert manager.started[-1][0] == 'robot_self_check'
    assert '--check-only' in manager.started[-1][1]
    assert 'id="startRobotButton"' in client.get('/').text
    assert 'id="robotStartupReport"' in client.get('/').text


def test_system_page_uses_content_height(tmp_path):
    client, _ = make_client(tmp_path)
    script = client.get('/static/app.js').text
    css = client.get('/static/styles.css').text
    page = client.get('/').text
    assert '["il", "rl", "rlt", "data", "system"].includes(name)' in script
    assert 'body.document-scroll .system-layout { height: auto; align-items: start; }' in css
    assert 'body.document-scroll .system-layout > .panel' in css
    assert 'id="robotStartupReport" style=' not in page
    assert 'class="panel robot-startup-panel"' in page
    assert '.system-layout .readiness-panel { min-height: 0; align-items: center; flex-direction: row;' in css


def test_robot_startup_refuses_active_recording(tmp_path):
    client, manager = make_client(tmp_path)
    manager.jobs = [{'id':'rec', 'kind':'record', 'state':'running'}]
    response = client.post('/api/system/start-robot', json={})
    assert response.status_code == 409
    assert manager.started == []


def test_groot_model_server_uses_groot_environment_and_pc(tmp_path, monkeypatch):
    client, manager = make_client(tmp_path)
    config = client.app.state.dashboard_config
    root = config.checkpoint_root / "groot_run"
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"architectures": ["Gr00tN1d6"]}))
    (root / "model.safetensors.index.json").write_text("{}")
    monkeypatch.setattr(
        web_dashboard,
        "_policy_server_status",
        lambda _config, _manager: {
            "ready": False,
            "state": "offline",
            "endpoint": "tcp://127.0.0.1:1",
        },
    )

    started = client.post(
        "/api/policy-server/start",
        json={"policy_id": "groot_run", "device": "cuda:1", "compute": "pc"},
    )

    assert started.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "policy_server"
    assert command[:3] == [str(config.groot_python), "-m", "cobot_bridge.groot_policy_server"]
    assert command[command.index("--model-path") + 1] == str(root)

    denied = client.post(
        "/api/policy-server/start",
        json={"policy_id": "groot_run", "device": "cuda:0", "compute": "robot"},
    )
    assert denied.status_code == 400
    assert "does not fit" in denied.json()["detail"]


def test_model_server_and_rollout_are_separate_jobs(tmp_path, monkeypatch):
    client, manager = make_client(tmp_path)
    monkeypatch.setattr(
        web_dashboard,
        "_policy_server_status",
        lambda _config, _manager: {
            "ready": False,
            "state": "offline",
            "endpoint": "tcp://127.0.0.1:1",
        },
    )

    started = client.post(
        "/api/policy-server/start",
        json={"policy_id": "policy", "device": "cuda:0", "num_steps": 5},
    )

    assert started.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "policy_server"
    assert command[:3] == ["/test/python", "-m", "cobot_bridge.policy_server"]
    assert command[command.index("--num-steps") + 1] == "5"
    assert "--bind" in command
    assert "--cameras" in command
    assert "--n-action-steps" not in command
    assert "--send-actions" not in command
    assert "--dataset-root" not in command
    assert "--dataset-repo-id" not in command

    robot_started = client.post(
        "/api/policy-server/start",
        json={"policy_id": "policy", "device": "cuda:0", "compute": "robot", "num_steps": 20},
    )
    assert robot_started.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "policy_server"
    assert command[:3] == ["/test/python", "-m", "cobot_bridge.remote_policy_server"]
    assert command[command.index("--ssh-host") + 1] == "agilex@192.168.250.2"
    assert command[command.index("--policy-id") + 1] == "policy"
    assert command[command.index("--num-steps") + 1] == "20"
    assert manager.jobs[0]["metadata"]["compute"] == "robot"


    monkeypatch.setattr(
        web_dashboard,
        "_policy_server_status",
        lambda _config, _manager: {
            "ready": True,
            "state": "ready",
            "policy_id": "policy",
        },
    )
    rollout = client.post(
        "/api/jobs/policy",
        json={
            "task": "pick object",
            "fps": 15,
            "mode": 1,
            "execute": True,
            "confirm_motion": True,
        },
    )

    assert rollout.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "policy_execute"
    assert "--policy-server-endpoint" in command
    assert "--startup-init-hold" in command
    assert "--policy-path" not in command
    assert "--dataset-root" not in command
    assert "--dataset-repo-id" not in command
    assert command[command.index("--fps") + 1] == "15.0"
    assert command[command.index("--mode") + 1] == "1"
    assert "--n-action-steps" not in command
    assert manager.jobs[0]["metadata"]["n_action_steps"] == "model_chunk"
    assert manager.jobs[0]["metadata"]["runner"] == "standard"

    explicit_rollout = client.post(
        "/api/jobs/policy",
        json={
            "task": "pick object",
            "fps": 15,
            "mode": 1,
            "n_action_steps": 12,
            "execute": True,
            "confirm_motion": True,
        },
    )

    assert explicit_rollout.status_code == 200
    _, explicit_command = manager.started[-1]
    assert explicit_command[explicit_command.index("--n-action-steps") + 1] == "12"

    smooth_rollout = client.post(
        "/api/jobs/policy",
        json={
            "task": "pick object",
            "fps": 30,
            "mode": 1,
            "runner": "smooth_v2",
            "execute": True,
            "confirm_motion": True,
        },
    )

    assert smooth_rollout.status_code == 200
    _, smooth_command = manager.started[-1]
    assert smooth_command[1:3] == ["-m", "cobot_bridge.run_lerobot_policy_smooth"]
    assert "--diagnostic-output" in smooth_command
    assert manager.jobs[0]["metadata"]["runner"] == "smooth_v2"
    assert manager.jobs[0]["metadata"]["diagnostic_output"].endswith(".jsonl")

    smooth_v3_rollout = client.post(
        "/api/jobs/policy",
        json={
            "task": "pick object",
            "fps": 30,
            "mode": 1,
            "runner": "smooth_v3",
            "execute": True,
            "confirm_motion": True,
        },
    )

    assert smooth_v3_rollout.status_code == 200
    _, smooth_v3_command = manager.started[-1]
    assert smooth_v3_command[1:3] == [
        "-m",
        "cobot_bridge.run_lerobot_policy_smooth_v3",
    ]

    edge_v4_rollout = client.post(
        "/api/jobs/policy",
        json={
            "task": "pick object",
            "fps": 30,
            "mode": 1,
            "runner": "edge_v4",
            "execute": True,
            "confirm_motion": True,
        },
    )

    assert edge_v4_rollout.status_code == 200
    _, edge_v4_command = manager.started[-1]
    assert edge_v4_command[1:3] == [
        "-m",
        "cobot_bridge.run_lerobot_policy_edge_v4",
    ]

    sync_rollout = client.post(
        "/api/jobs/policy",
        json={
            "task": "pick object",
            "fps": 15,
            "mode": 2,
            "execute": True,
            "confirm_motion": True,
        },
    )

    assert sync_rollout.status_code == 200
    _, sync_command = manager.started[-1]
    assert sync_command[sync_command.index("--mode") + 1] == "2"
    assert manager.jobs[0]["metadata"]["mode"] == 2


def test_guided_rtc_server_owns_terminal_and_matches_checkpoint_action_space(tmp_path, monkeypatch):
    client, manager = make_client(tmp_path)
    config = client.app.state.dashboard_config
    monkeypatch.setattr(
        web_dashboard,
        "_policy_server_status",
        lambda _config, _manager: {
            "ready": False,
            "state": "offline",
            "endpoint": "tcp://127.0.0.1:1",
        },
    )

    started = client.post(
        "/api/policy-server/start",
        json={
            "policy_id": "policy",
            "device": "cuda:0",
            "compute": "pc",
            "inference_chunking": "guided_rtc",
        },
    )

    assert started.status_code == 200
    _, server_command = manager.started[-1]
    assert server_command[:3] == [str(config.python), "-m", "cobot_bridge.policy_server"]
    assert server_command[server_command.index("--inference-chunking") + 1] == "guided_rtc"
    # One terminal target is a one-point prefix, not a 10-step overlap.
    assert server_command[server_command.index("--rtc-execution-horizon") + 1] == "1"

    monkeypatch.setattr(
        web_dashboard,
        "_policy_server_status",
        lambda _config, _manager: {
            "ready": True,
            "state": "ready",
            "compute": "pc",
            "endpoint": "tcp://127.0.0.1:5560",
            "policy_id": "policy",
            "inference_chunking": "guided_rtc",
            "relative_eef_actions": True,
        },
    )
    wrong_space = client.post(
        "/api/jobs/policy",
        json={
            "task": "pick object",
            "mode": 1,
            "runner": "guided_rtc",
            "action_space": "joint",
        },
    )
    assert wrong_space.status_code == 409
    assert "relative EEF" in wrong_space.json()["detail"]

    rollout = client.post(
        "/api/jobs/policy",
        json={
            "task": "pick object",
            "mode": 1,
            "runner": "guided_rtc",
            "action_space": "eef_target",
        },
    )
    assert rollout.status_code == 200
    _, rollout_command = manager.started[-1]
    assert rollout_command[1:3] == ["-m", "cobot_bridge.run_lerobot_policy"]
    assert "--guided-rtc-terminal" in rollout_command
    assert "--n-action-steps" not in rollout_command
    assert "rtc_leftover_targets" not in rollout_command


def test_robot_gpu_server_runs_rollout_pipeline_on_robot(tmp_path, monkeypatch):
    client, manager = make_client(tmp_path)
    monkeypatch.setattr(
        web_dashboard,
        "_policy_server_status",
        lambda _config, _manager: {
            "ready": True,
            "state": "ready",
            "compute": "robot",
            "endpoint": "tcp://192.168.250.2:5560",
            "policy_id": "policy",
        },
    )

    response = client.post(
        "/api/jobs/policy",
        json={
            "task": "pick object",
            "fps": 15,
            "mode": 2,
            "execute": True,
            "confirm_motion": True,
        },
    )

    assert response.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "policy_execute"
    assert command[:3] == ["/test/python", "-m", "cobot_bridge.remote_rollout"]
    assert command[command.index("--ssh-host") + 1] == "agilex@192.168.250.2"
    assert command[command.index("--endpoint") + 1] == "tcp://127.0.0.1:1"
    assert command[command.index("--policy-server-endpoint") + 1] == "tcp://127.0.0.1:5560"
    assert command[command.index("--runner") + 1] == "standard"
    assert "--send-actions" in command
    assert "--startup-init-hold" in command
    assert manager.jobs[0]["metadata"]["compute"] == "robot"


def test_init_pose_requires_confirmation_and_uses_fixed_adapter(tmp_path):
    client, manager = make_client(tmp_path)

    denied = client.post(
        "/api/jobs/init-pose",
        json={"execute": True, "confirm_motion": False},
    )
    assert denied.status_code == 400

    allowed = client.post(
        "/api/jobs/init-pose",
        json={"execute": True, "confirm_motion": True},
    )
    assert allowed.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "init_pose_execute"
    assert command[:3] == ["/test/python", "-m", "cobot_bridge.move_init_pose"]
    assert command[command.index("--joint-speed") + 1] == "0.48"
    assert command[command.index("--min-seconds") + 1] == "1.0"
    assert "--execute" in command


def test_il_init_and_record_requires_confirmation_and_sequences_commands(tmp_path):
    client, manager = make_client(tmp_path)
    monitor = FakeDeviceMonitor()
    client = TestClient(create_app(client.app.state.dashboard_config, manager, monitor))
    payload = {
        "collection": "il_init_record",
        "task": "pick object",
        "side": "right",
        "cameras": ["front", "right"],
        "fps": 30,
        "stop_on_teach_release": True,
        "collection_type": "IL",
    }

    denied = client.post(
        "/api/jobs/init-and-record",
        json={**payload, "confirm_motion": False},
    )
    assert denied.status_code == 400
    assert manager.started == []

    allowed = client.post(
        "/api/jobs/init-and-record",
        json={**payload, "confirm_motion": True},
    )
    assert allowed.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "record_with_init"
    assert command[:3] == ["/test/python", "-m", "cobot_bridge.init_then_record"]
    separator = command.index("--")
    record_arguments = command[separator + 1 :]
    assert record_arguments[record_arguments.index("--collection-type") + 1] == "IL"
    assert "--stop-on-teach-release" in record_arguments
    assert manager.jobs[0]["metadata"]["phase"] == "queued"


def test_emergency_and_exit_stop_use_dedicated_bridge_commands(tmp_path, monkeypatch):
    client, manager = make_client(tmp_path)
    calls = []

    class FakeControlClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def emergency_safe(self):
            calls.append("emergency")
            return {"accepted": True, "emergency_latched": True}

        def clear_emergency(self):
            calls.append("clear")
            return {"accepted": True, "emergency_latched": False}

    monkeypatch.setattr(web_dashboard, "CobotBridgeClient", FakeControlClient)

    stopped = client.post("/api/emergency-safe")
    assert stopped.status_code == 200
    assert manager.motion_stopped is True
    cleared = client.post("/api/clear-emergency")
    assert cleared.status_code == 200
    assert calls == ["emergency", "clear"]


def test_configured_roots_cannot_be_escaped(tmp_path):
    client, manager = make_client(tmp_path)

    response = client.post(
        "/api/jobs/convert",
        json={"raw_id": "../../etc/passwd", "dataset_name": "safe_name"},
    )

    assert response.status_code == 400
    assert manager.started == []


def test_rlt_tab_inventory_and_read_only_jobs(tmp_path, monkeypatch):
    client, manager = make_client(tmp_path)
    config = client.app.state.dashboard_config
    client = TestClient(create_app(config, manager, FakeDeviceMonitor()))
    stage1 = config.rlt_root / "checkpoints" / "stage1_run"
    stage1.mkdir(parents=True)
    (stage1 / "vla_best").mkdir()
    base_policy = tmp_path / "base_policy"
    base_policy.mkdir()
    (stage1 / "best_checkpoint.pt").write_bytes(b"checkpoint")
    (stage1 / "stage1_config.json").write_text(
        json.dumps(
            {
                "stage": "rlt_stage1_joint",
                "rlt_architecture": "rlinf_prefix_v1",
                "smolvla_path": str(base_policy),
                "dataset_root": str(tmp_path / "dataset"),
                "dataset_frames": 36949,
                "dataset_episodes": 79,
                "dataset_fps": 30,
                "prefix_shape": [177, 960],
                "rlt_config": {"rlt_hidden_dim": 256, "chunk_size": 30},
            }
        )
    )
    (stage1 / "loss_history.json").write_text(
        json.dumps([{"step": 5000, "loss": 0.2796}])
    )
    actor_run = config.rlt_root / "checkpoints" / "stage1_run_actor_bc"
    actor_run.mkdir()
    actor_seed = actor_run / "checkpoint_00000000.pt"
    torch.save(
        {
            "stage": "rlt_stage2",
            "actor_output_semantics": web_dashboard.ACTOR_OUTPUT_SEMANTICS,
            "action_chunk_timebase": web_dashboard.ACTION_CHUNK_TIMEBASE,
            "stage1_checkpoint": str((stage1 / "best_checkpoint.pt").resolve()),
            "ready_for_online": True,
            "actor_state_dict": {},
            "total_updates": 0,
        },
        actor_seed,
    )
    (actor_run / "status.json").write_text(
        json.dumps(
            {
                "stage": "rlt_stage2",
                "actor_output_semantics": web_dashboard.ACTOR_OUTPUT_SEMANTICS,
                "action_chunk_timebase": web_dashboard.ACTION_CHUNK_TIMEBASE,
                "initialization": "il_ground_truth_actor_bc",
                "bc_steps": 19750,
                "validation_gt_bc_loss": 0.0085327,
                "total_updates": 0,
                "ready_for_online": True,
                "stage1_checkpoint": str((stage1 / "best_checkpoint.pt").resolve()),
                "checkpoint": str(actor_seed.resolve()),
                "latest": str((actor_run / "latest.pt").resolve()),
            }
        )
    )
    steam_actor = config.rlt_root / "checkpoints" / "stage1_run_steam_offline_actor_20k.pt"
    torch.save(
        {
            "stage": web_dashboard.STEAM_OFFLINE_ACTOR_STAGE,
            "actor_output_semantics": web_dashboard.ACTOR_OUTPUT_SEMANTICS,
            "action_chunk_timebase": web_dashboard.ACTION_CHUNK_TIMEBASE,
            "stage1_checkpoint": str((stage1 / "best_checkpoint.pt").resolve()),
            "actor_state_dict": {},
            "step": 1,
        },
        steam_actor,
    )
    offline_run = config.rlt_root / "checkpoints" / "stage1_run_stage2_offline_eval"
    offline_run.mkdir()
    for updates in (2000, 5000):
        (offline_run / f"checkpoint_{updates:08d}.pt").write_bytes(b"candidate")
    (offline_run / "actor_inventory.json").write_text(
        json.dumps(
            {
                "format": "cobot_magic_rlt_actor_inventory_v1",
                "publish_scope": "manual_evaluation_only",
                "actor_output_semantics": web_dashboard.ACTOR_OUTPUT_SEMANTICS,
                "action_chunk_timebase": web_dashboard.ACTION_CHUNK_TIMEBASE,
                "stage1_checkpoint": str((stage1 / "best_checkpoint.pt").resolve()),
                "candidates": [
                    {
                        "checkpoint": "checkpoint_00002000.pt",
                        "total_updates": 2000,
                        "label": "Offline Full234 · 2k",
                    },
                    {
                        "checkpoint": "checkpoint_00005000.pt",
                        "total_updates": 5000,
                        "label": "Offline Full234 · 5k",
                    },
                ],
            }
        )
    )
    config.run_root.mkdir(parents=True, exist_ok=True)
    trainer_progress = config.run_root / "rlt_actor_mode1_test.pipeline.trainer.json"
    trainer_progress.write_text(
        json.dumps(
            {
                "state": "running",
                "current_update": 40,
                "total_updates": 100,
                "percent": 40.0,
                "updates_per_second": 8.0,
                "eta_seconds": 7.5,
            }
        )
    )
    (config.run_root / "rlt_actor_mode1_test.pipeline.json").write_text(
        json.dumps(
            {
                "state": "running",
                "stage1_checkpoint": str((stage1 / "best_checkpoint.pt").resolve()),
                "trainer_progress_file": str(trainer_progress.resolve()),
                "rollout_phase": "actor",
            }
        )
    )

    # Stage-1 inventory is refreshed off the overview request path so polling
    # remains responsive while the checkpoint tree is scanned.
    artifacts = []
    for _ in range(100):
        overview = client.get("/api/overview")
        assert overview.status_code == 200
        artifacts = overview.json()["rlt"]["stage1_checkpoints"]
        if artifacts:
            break
        time.sleep(0.01)
    assert len(artifacts) == 1
    assert artifacts[0]["id"] == "stage1_run/best_checkpoint.pt"
    assert artifacts[0]["ready"] is True
    assert artifacts[0]["embedding_shapes"]["prefix"] == [177, 960]
    assert artifacts[0]["actor_checkpoints"][0]["kind"] == "pre_rl_bc"
    assert artifacts[0]["actor_checkpoints"][0]["total_updates"] == 0
    assert artifacts[0]["actor_checkpoints"][0]["bc_steps"] == 19750
    steam = [
        actor
        for actor in artifacts[0]["actor_checkpoints"]
        if actor["kind"] == "offline_steam"
    ]
    assert len(steam) == 1
    assert steam[0]["read_only"] is True
    assert steam[0]["inference_only"] is True
    assert "STEAM Offline Actor" in steam[0]["label"]
    offline = [
        actor
        for actor in artifacts[0]["actor_checkpoints"]
        if actor["kind"] == "offline_eval"
    ]
    assert [actor["total_updates"] for actor in offline] == [5000, 2000]
    assert all(actor["follows_latest"] is False for actor in offline)
    assert artifacts[0]["checkpoint_path"] == str(
        (stage1 / "best_checkpoint.pt").resolve()
    )
    assert artifacts[0]["stage2"]["pipeline"]["trainer_progress"]["current_update"] == 40
    assert overview.json()["rlt"]["online_stage2_ready"] is True

    validated = client.post(
        "/api/jobs/rlt-validate",
        json={"checkpoint_id": "stage1_run/best_checkpoint.pt"},
    )
    assert validated.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "rlt_validate"
    assert command[:3] == ["/test/python", "-m", "cobot_bridge.rlt_validate"]
    assert "--rlt-checkpoint" in command
    assert manager.jobs[0]["metadata"]["motion"] is False

    manager.jobs = []
    shadow = client.post(
        "/api/jobs/rlt-shadow",
        json={
            "checkpoint_id": "stage1_run/best_checkpoint.pt",
            "task": "pick cross",
            "side": "right",
            "fps": 30,
            "duration": 0,
            "device": "cuda:0",
        },
    )
    assert shadow.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "rlt_shadow"
    assert command[:3] == ["/test/python", "-m", "cobot_bridge.rlt_shadow"]
    assert command[command.index("--fps") + 1] == "30.0"
    assert command[command.index("--duration") + 1] == "0.0"
    assert "--send-actions" not in command
    assert "--execute" not in command
    assert manager.jobs[0]["metadata"]["send_actions"] is False
    assert manager.jobs[0]["metadata"]["duration"] == 0

    manager.jobs = []
    server = client.post(
        "/api/rlt-policy-server/start",
        json={
            "checkpoint_id": "stage1_run/best_checkpoint.pt",
            "actor_checkpoint_id": "stage1_run_actor_bc/checkpoint_00000000.pt",
            "device": "cuda:0",
        },
    )
    assert server.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "rlt_policy_server"
    assert command[:3] == ["/test/python", "-m", "cobot_bridge.rlt_policy_server"]
    assert command[command.index("--stage1-checkpoint") + 1].endswith(
        "stage1_run/best_checkpoint.pt"
    )
    assert command[command.index("--stage2-checkpoint") + 1].endswith(
        "stage1_run_actor_bc/checkpoint_00000000.pt"
    )
    assert manager.jobs[0]["metadata"]["actor_checkpoint_id"] == (
        "stage1_run_actor_bc/checkpoint_00000000.pt"
    )
    assert manager.jobs[0]["metadata"]["motion"] is False

    monkeypatch.setattr(
        web_dashboard,
        "_rlt_policy_server_status",
        lambda _config, _manager: {
            "ready": True,
            "state": "ready",
            "checkpoint_id": "stage1_run/best_checkpoint.pt",
            "stage1_checkpoint": str(stage1 / "best_checkpoint.pt"),
            "device": "cuda:0",
            "actor_generation": 0,
            "actor_ready": False,
            "active_episode": False,
        },
    )

    manager.jobs = []
    denied = client.post(
        "/api/jobs/rlt-online",
        json={
            "checkpoint_id": "stage1_run/best_checkpoint.pt",
            "task": "pick cross",
            "workflow_mode": 1,
            "rollout_phase": "warmup",
            "execute": True,
            "confirm_motion": False,
        },
    )
    assert denied.status_code == 422

    actor_denied = client.post(
        "/api/jobs/rlt-online",
        json={
            "checkpoint_id": "stage1_run/best_checkpoint.pt",
            "task": "pick cross",
            "workflow_mode": 1,
            "rollout_phase": "actor",
            "collection": "test_rlt",
            "execute": True,
            "confirm_motion": True,
        },
    )
    assert actor_denied.status_code == 409
    assert "actor is not ready" in actor_denied.json()["detail"]

    reference_online = client.post(
        "/api/jobs/rlt-online",
        json={
            "checkpoint_id": "stage1_run/best_checkpoint.pt",
            "task": "pick cross",
            "workflow_mode": 1,
            "rollout_phase": "reference",
            "collection": "test_rlt",
            "actor_exploration_std": 0.05,
            "execute": True,
            "confirm_motion": True,
        },
    )
    assert reference_online.status_code == 200
    reference_command = manager.started[-1][1]
    assert reference_command[reference_command.index("--rollout-phase") + 1] == "reference"
    assert reference_command[reference_command.index("--actor-exploration-std") + 1] == "0.0"
    assert reference_command[reference_command.index("--rollout-speed") + 1] == "0.5"
    assert "--replan-actions" not in reference_command
    assert "--startup-init-hold" not in reference_command
    assert "--enable-actor" not in reference_command
    assert reference_command[reference_command.index("--actor-max-decisions") + 1] == "0"
    assert reference_online.json()["actor_enabled"] is False
    assert manager.jobs[-1]["metadata"]["training_requested"] is False
    manager.jobs = []

    # A compatible Actor explicitly selected for the persistent server is
    # rollout-ready even when the canonical <stage1>_stage2/latest.pt alias has
    # not been published.
    monkeypatch.setattr(
        web_dashboard,
        "_rlt_policy_server_status",
        lambda _config, _manager: {
            "ready": True,
            "state": "ready",
            "checkpoint_id": "stage1_run/best_checkpoint.pt",
            "actor_checkpoint_id": "stage1_run_actor_bc/checkpoint_00000000.pt",
            "stage1_checkpoint": str(stage1 / "best_checkpoint.pt"),
            "device": "cuda:0",
            "actor_generation": 0,
            "actor_ready": False,
            "active_episode": False,
        },
    )
    selected_actor_online = client.post(
        "/api/jobs/rlt-online",
        json={
            "checkpoint_id": "stage1_run/best_checkpoint.pt",
            "task": "pick cross",
            "workflow_mode": 1,
            "rollout_phase": "actor",
            "collection": "test_rlt",
            "actor_exploration_std": 0.0,
            "actor_eef_gate": False,
            "smooth_profile": "smooth_v3",
            "execute": True,
            "confirm_motion": True,
        },
    )
    # A selected Actor without a replay lineage remains executable for
    # controlled data collection, but must not silently start async updates.
    assert selected_actor_online.status_code == 200
    assert selected_actor_online.json()["inference_only"] is True
    selected_actor_command = manager.started[-1][1]
    assert (
        selected_actor_command[
            selected_actor_command.index("--actor-max-decisions") + 1
        ]
        == "1"
    )
    assert manager.jobs[-1]["metadata"]["training_requested"] is False
    manager.jobs = []

    stage2 = stage1.with_name("stage1_run_stage2")
    stage2.mkdir()
    (stage2 / "latest.pt").write_bytes(b"current actor")
    (stage2 / "status.json").write_text(
        json.dumps(
            {
                "stage": "rlt_stage2",
                "actor_output_semantics": web_dashboard.ACTOR_OUTPUT_SEMANTICS,
                "action_chunk_timebase": web_dashboard.ACTION_CHUNK_TIMEBASE,
                "stage1_checkpoint": str((stage1 / "best_checkpoint.pt").resolve()),
                "total_updates": 100,
                "ready_for_online": True,
                "latest": str((stage2 / "latest.pt").resolve()),
            }
        )
    )

    old_stage1 = config.rlt_root / "checkpoints" / "old_stage1_run"
    old_stage1.mkdir()
    old_base = tmp_path / "old_base_policy"
    old_base.mkdir()
    old_checkpoint = old_stage1 / "best_checkpoint.pt"
    old_checkpoint.write_bytes(b"old stage1")
    (old_stage1 / "stage1_config.json").write_text(
        json.dumps(
            {
                "stage": "rlt_stage1",
                "smolvla_path": str(old_base),
                "dataset_root": str(tmp_path / "old_dataset"),
                "dataset_frames": 36949,
                "dataset_episodes": 79,
                "dataset_fps": 30,
                "embedding_shapes": {"vlm": [177, 960], "expert": [30, 720]},
                "rlt_config": {"rlt_hidden_dim": 256, "chunk_size": 30},
            }
        )
    )
    (old_stage1 / "loss_history.json").write_text(
        json.dumps([{"step": 5000, "loss": 0.2796}])
    )
    old_stage2 = old_stage1.with_name("old_stage1_run_stage2")
    old_stage2.mkdir()
    warmup_actor = old_stage2 / "checkpoint_00030000.pt"
    torch.save(
        {
            "stage": "rlt_stage2",
            "actor_output_semantics": web_dashboard.ACTOR_OUTPUT_SEMANTICS,
            "action_chunk_timebase": web_dashboard.ACTION_CHUNK_TIMEBASE,
            "ready_for_online": True,
            "total_updates": 30000,
            "training_phase": "warmup",
            "stage1_checkpoint": str(old_checkpoint.resolve()),
            "actor_state_dict": {},
            "actor_action_scale": [1.0] * 7,
        },
        warmup_actor,
    )
    (old_stage2 / "latest.pt").write_bytes(b"published alias")
    (old_stage2 / "status.json").write_text(
        json.dumps(
            {
                "stage": "rlt_stage2",
                "actor_output_semantics": web_dashboard.ACTOR_OUTPUT_SEMANTICS,
                "action_chunk_timebase": web_dashboard.ACTION_CHUNK_TIMEBASE,
                "stage1_checkpoint": str(old_checkpoint.resolve()),
                "total_updates": 30000,
                "training_phase": "warmup",
                "ready_for_online": True,
                "checkpoint": str(warmup_actor.resolve()),
                "latest": str((old_stage2 / "latest.pt").resolve()),
            }
        )
    )
    refreshed = []
    old = None
    for _ in range(100):
        refreshed = client.get("/api/overview").json()["rlt"]["stage1_checkpoints"]
        old = next(
            (item for item in refreshed if item["id"] == "old_stage1_run/best_checkpoint.pt"),
            None,
        )
        if old is not None:
            break
        time.sleep(0.01)
    assert old is not None
    assert old["ready"] is True
    assert old["architecture"] == "legacy_action_conditioned_v2"
    assert old["inference_only"] is True
    assert old["embedding_shapes"] == {"vlm": [177, 960], "expert": [30, 720]}
    assert len(old["actor_checkpoints"]) == 1
    warmup = old["actor_checkpoints"][0]
    assert warmup["id"] == "old_stage1_run_stage2/latest.pt"
    assert warmup["kind"] == "warmup_only"
    assert warmup["label"] == "Warm-up-only Actor · 30k"
    assert warmup["follows_latest"] is True
    assert all(
        not actor["label"].startswith("Legacy")
        for item in refreshed
        for actor in item["actor_checkpoints"]
    )
    online = client.post(
        "/api/jobs/rlt-online",
        json={
            "checkpoint_id": "stage1_run/best_checkpoint.pt",
            "task": "pick cross",
            "workflow_mode": 1,
            "rollout_phase": "actor",
            "collection": "test_rlt",
            "actor_exploration_std": 0.0001,
            "execute": True,
            "confirm_motion": True,
        },
    )
    assert online.status_code == 200
    assert online.json()["inference_only"] is True
    assert manager.jobs[-1]["metadata"]["training_requested"] is False



def test_rlt_checkpoint_cannot_escape_configured_root(tmp_path):
    client, manager = make_client(tmp_path)

    response = client.post(
        "/api/jobs/rlt-shadow",
        json={"checkpoint_id": "../../outside.pt", "task": "pick cross"},
    )

    assert response.status_code == 400
    assert manager.started == []


def test_rlt_stage2_can_rollback_atomically_to_fixed_30k(tmp_path, monkeypatch):
    import torch

    client, manager = make_client(tmp_path)
    config = client.app.state.dashboard_config
    stage1 = config.rlt_root / "checkpoints" / "stage1_run"
    stage1.mkdir(parents=True)
    stage1_checkpoint = stage1 / "best_checkpoint.pt"
    stage1_checkpoint.write_bytes(b"stage1")
    stage2 = stage1.with_name("stage1_run_stage2")
    stage2.mkdir()
    target = stage2 / "checkpoint_00030000.pt"
    torch.save(
        {
            "format_version": 3,
            "stage": "rlt_stage2",
            "actor_output_semantics": web_dashboard.ACTOR_OUTPUT_SEMANTICS,
            "action_chunk_timebase": web_dashboard.ACTION_CHUNK_TIMEBASE,
            "stage1_checkpoint": str(stage1_checkpoint.resolve()),
            "total_updates": 30000,
            "ready_for_online": True,
            "buffer_transitions": 1876,
            "actor_updates_this_run": 7500,
            "training_phase": "warmup",
            "actor_exploration_std": 0.002,
            "actor_action_scale": [1.0] * 7,
            "actor_validation": {"mean_abs_error_to_reference": 0.02},
            "actor_warnings": [],
            "last_metrics": {"critic_loss": 0.01},
            "history": [{"updates": 30000, "new_transitions": 1876}],
            "trained_replay_names": ["base.npz"],
        },
        target,
    )
    replay_root = config.rlt_root / "replay" / "stage1_run"
    replay_root.mkdir(parents=True)
    (replay_root / "base.npz").write_bytes(b"base")
    (replay_root / "new_failure.npz").write_bytes(b"bad replay")
    (replay_root / "new_failure.npz.json").write_text("{}")
    (replay_root / "new_failure.decisions.h5").write_bytes(b"decisions")
    latest = stage2 / "latest.pt"
    latest.write_bytes(b"newer actor")
    (stage2 / "status.json").write_text(
        json.dumps(
            {
                "stage": "rlt_stage2",
                "actor_output_semantics": web_dashboard.ACTOR_OUTPUT_SEMANTICS,
                "action_chunk_timebase": web_dashboard.ACTION_CHUNK_TIMEBASE,
                "stage1_checkpoint": str(stage1_checkpoint.resolve()),
                "total_updates": 35445,
                "ready_for_online": True,
            }
        )
    )
    monkeypatch.setattr(
        web_dashboard,
        "_rlt_policy_server_status",
        lambda _config, _manager: {"ready": False, "active_episode": False},
    )

    response = client.post(
        "/api/rlt-stage2/rollback-30k",
        json={
            "checkpoint_id": "stage1_run/best_checkpoint.pt",
            "target_updates": 30000,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["rolled_back"] is True
    assert result["previous_updates"] == 35445
    assert result["target_updates"] == 30000
    assert result["motion"] is False
    assert result["quarantined_replays"] == 1
    assert latest.read_bytes() == target.read_bytes()
    status = json.loads((stage2 / "status.json").read_text())
    assert status["total_updates"] == 30000
    assert status["rollback_from_total_updates"] == 35445
    backups = list((stage2 / "rollback_archive").glob("latest_before_rollback_*.pt"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"newer actor"
    quarantine = Path(result["quarantine"])
    assert (quarantine / "new_failure.npz").read_bytes() == b"bad replay"
    assert (quarantine / "new_failure.npz.json").is_file()
    assert (quarantine / "new_failure.decisions.h5").is_file()
    assert (replay_root / "base.npz").is_file()
    assert not (replay_root / "new_failure.npz").exists()
    assert manager.started == []


def test_convert_accepts_hybrid_il_action_rule_and_trim(tmp_path):
    client, manager = make_client(tmp_path)

    response = client.post(
        "/api/jobs/convert",
        json={
            "raw_id": "episode.raw.h5",
            "dataset_name": "hybrid_il",
            "action_source": "puppet_state_t3_arm_cmd_gripper",
            "trim_stationary_holds": True,
            "use_videos": True,
        },
    )

    assert response.status_code == 200
    kind, command = manager.started[-1]
    assert kind == "convert"
    assert command[command.index("--action-source") + 1] == (
        "puppet_state_t3_arm_cmd_gripper"
    )
    assert "--trim-stationary-holds" in command


def test_tcp_endpoint_probe_detects_a_live_holder():
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        endpoint = f"tcp://127.0.0.1:{listener.getsockname()[1]}"
        assert web_dashboard._tcp_endpoint_in_use(endpoint)
    finally:
        listener.close()
    assert not web_dashboard._tcp_endpoint_in_use(endpoint)


def test_tcp_endpoint_probe_ignores_non_tcp_transports():
    assert not web_dashboard._tcp_endpoint_in_use("ipc:///tmp/rlt-policy-server")
    assert not web_dashboard._tcp_endpoint_in_use("inproc://whatever")


def test_actor_frame_gate_converts_to_complete_reference_decisions():
    assert web_dashboard._actor_takeover_decisions_for_frame(960, 30, 0.5) == 16
    assert web_dashboard._actor_takeover_decisions_for_frame(780, 30, 0.5) == 13
    assert web_dashboard._actor_takeover_decisions_for_frame(None, 30, 0.5) == 0


def test_v7_actor_uses_deployment_handover_at_decision_9():
    checkpoint = Path(
        "/tmp/checkpoints/"
        "cobot_magic_smolvla_il2trimrel80k_rlt_stage1_rlinf960_ref30_actor30_5k_"
        "bs64_stage2_actor20_speed05_seq_m26q1_ilfakebc_v1v4q1_utd5_g720_loose_"
        "v1_frozen83_v7/latest.pt"
    )
    assert (
        web_dashboard._actor_takeover_decisions_for_checkpoint(
            checkpoint, 720, 30, 0.5
        )
        == 9
    )


def test_other_actor_keeps_checkpoint_frame_gate():
    checkpoint = Path("/tmp/checkpoints/other_actor/latest.pt")
    assert (
        web_dashboard._actor_takeover_decisions_for_checkpoint(
            checkpoint, 720, 30, 0.5
        )
        == 12
    )
