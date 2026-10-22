"""Inspect and start missing Dashboard dependencies; never enable robot output.

Runs on the Dashboard host. All AGX changes are explicit, component-scoped
tmux launches. Existing but unhealthy processes are not killed or duplicated.
"""
from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path

from .client import CobotBridgeClient
from .restart_devices import (
    CAMERAS, CAN_INTERFACES, PREVIEW_PORT, PREVIEW_PYTHON,
    DeviceRestartError, _camera_command, _ensure_can,
    _ensure_realsense_cameras, _ssh, _tmux_shell,
)


def remote_inventory(host: str, workspace: Path) -> dict:
    # Inspect actual processes, not tmux session existence. No ROS CLI daemon,
    # camera enumeration, password, or mutation in this probe.
    code = r'''
import json, subprocess, urllib.request
from pathlib import Path
processes=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        argv=p.read_bytes().decode(errors='replace').strip('\0').split('\0')
        exe=Path(argv[0]).name
        if exe not in ('bash','sh','ssh','sshd','tmux','sudo','timeout'):
            processes.append(argv)
    except (OSError,IndexError): pass
def module(name):
    return any(any(a[i:i+2]==['-m',name] for i in range(len(a)-1)) for a in processes)
links=json.loads(subprocess.check_output(['ip','-j','link'],text=True))
can={x['ifname']: 'UP' in x.get('flags',[]) for x in links}
try:
    with urllib.request.urlopen('http://127.0.0.1:8091/health',timeout=2) as r:
        preview=json.load(r)
except Exception: preview={}
print(json.dumps(dict(
    can=can,
    arm_nodes=sum(any(Path(v).name=='piper_single_ctrl' for v in a[:2]) for a in processes),
    arm_launch=any(any(Path(v).name=='ros2' for v in a[:2]) and 'launch' in a and any('cobot_magic_master_puppet.launch.py' in v for v in a) for a in processes),
    arbiter=any(any(Path(v).name=='cobot_magic_robot_control.py' for v in a[1:]) for a in processes),
    camera_nodes=sum('realsense2_camera_node' in Path(a[0]).name for a in processes),
    camera_launch=any(any(Path(v).name=='ros2' for v in a[:2]) and 'launch' in a and any('cobot_magic_mixed_camera.launch.py' in v for v in a) for a in processes),
    bridge_process=module('cobot_bridge.server'),
    preview_process=module('cobot_bridge.cobot_mjpeg_cam_server'),
    preview=preview,
    external_activity=any(any(v.startswith('cobot_bridge.run_lerobot_policy') or v in ('cobot_bridge.record_raw','cobot_bridge.replay_raw','cobot_bridge.lerobot_replay','cobot_bridge.run_rlt_policy') for v in a) for a in processes),
)))
'''
    result = _ssh(host, ['python3', '-c', code])
    inventory = json.loads(result.stdout)
    required = ['cobot_bridge/start_robot_bridge.sh',
                'scripts/cobot_magic_master_puppet.launch.py',
                'scripts/cobot_magic_mixed_camera.launch.py']
    _ssh(host, ['test', '-d', str(workspace)])
    for relative in required:
        _ssh(host, ['test', '-f', str(workspace / relative)])
    return inventory


def bridge_inventory(endpoint: str) -> dict:
    try:
        with CobotBridgeClient(endpoint, timeout_s=1.5) as client:
            raw, images = client.get_observation(sides=('left', 'right'), images=CAMERAS)
        arms = {f'{role}_{side}': bool((s := (raw.get(role) or {}).get(side))
                and s.get('fresh') and len(s.get('position', [])) == 7)
                for role in ('master', 'puppet') for side in ('left', 'right')}
        cameras = {name: bool(((raw.get('images') or {}).get(name) or {}).get('fresh')
                             and images.get(name)) for name in CAMERAS}
        return dict(online=True, arms=arms, cameras=cameras, control=raw.get('control') or {})
    except Exception as exc:
        return dict(online=False, error=str(exc), arms={}, cameras={}, control={})


def assert_idle(bridge: dict, remote: dict) -> None:
    c = bridge.get('control', {})
    if remote.get('external_activity') or c.get('teach_active') or c.get('output_enabled'):
        raise DeviceRestartError('请先结束录制/rollout、关闭 Teach 和运动输出；一键启动不会中断现有操作。')
    if remote.get('arm_nodes', 0) and not (
        bridge.get('online') and c.get('effective_mode') == 'SAFE'
        and c.get('output_enabled') is False and c.get('teach_active') is False
    ):
        raise DeviceRestartError('现有机械臂的 SAFE/Teach 状态无法确认，停止补启动；请人工检查。')


def missing_components(remote: dict, bridge: dict) -> list[str]:
    missing = []
    if not all(remote['can'].get(n, False) for n in CAN_INTERFACES):
        missing.append('can')
    if not bridge.get('online'):
        missing.append('bridge')
    if not all(bridge.get('arms', {}).get(f'{r}_{s}', False)
               for r in ('master', 'puppet') for s in ('left', 'right')):
        missing.append('arms')
    if not all(bridge.get('cameras', {}).get(n, False) for n in CAMERAS):
        missing.append('cameras')
    health = remote.get('preview', {})
    if not all(isinstance(health.get(n), dict) and health[n].get('version', 0) > 0
               for n in CAMERAS):
        missing.append('preview')
    return missing


def start_component(host: str, workspace: Path, component: str) -> None:
    q = shlex.quote
    if component == 'bridge':
        command = f'export COBOT_BRIDGE_BIND=tcp://0.0.0.0:5555; exec bash {q(str(workspace / "cobot_bridge/start_robot_bridge.sh"))}'
    elif component == 'arms':
        command = (
            'source /opt/ros/humble/setup.bash && '
            'source /home/agilex/piper_ros/install/setup.bash && '
            'unset ROS_DOMAIN_ID; export ROS_LOCALHOST_ONLY=1; '
            f'cd {q(str(workspace))} && exec ros2 launch '
            'scripts/cobot_magic_master_puppet.launch.py live_output:=true'
        )
    elif component == 'cameras':
        command = _camera_command(workspace)
    elif component == 'preview':
        command = (f'cd {q(str(workspace))} && exec {q(PREVIEW_PYTHON)} '
                   '-m cobot_bridge.cobot_mjpeg_cam_server '
                   f'--port {PREVIEW_PORT} --bridge-endpoint tcp://127.0.0.1:5555')
    else:
        raise ValueError(component)
    # Keep the established session/window names so the existing explicit
    # Restart All Devices operation can still find and stop these drivers.
    session, window = {
        'bridge': ('cobot_bridge', 'bridge'), 'arms': ('cobot_magic', 'arms'),
        'cameras': ('cobot_magic', 'camera'), 'preview': ('cobot_collect', 'preview'),
    }[component]
    exists = _ssh(host, ['tmux', 'has-session', '-t', session], check=False).returncode == 0
    if exists:
        args = ['tmux', 'new-window', '-d', '-t', session, '-n', window]
        windows = _ssh(host, ['tmux', 'list-windows', '-t', session, '-F', '#{window_name}']).stdout.splitlines()
        if window in windows:
            raise DeviceRestartError(f'{session}:{window} 已存在但未就绪；请查看该窗口，不重复启动。')
    else:
        args = ['tmux', 'new-session', '-d', '-s', session, '-n', window]
    _ssh(host, args + [_tmux_shell(command)])


def ensure_robot(host: str, workspace: Path, endpoint: str, report_path: Path,
                 *, check_only=False, timeout_s=90., sudo_password_file=None) -> dict:
    report = dict(state='running', phase='self_check', actions=[], started_at_ns=time.time_ns())

    def save(phase, **values):
        report.update(phase=phase, **values)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = report_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        tmp.replace(report_path)
        print(json.dumps(dict(phase=phase, **values), ensure_ascii=False), flush=True)

    try:
        save('self_check')
        remote = remote_inventory(host, workspace)
        bridge = bridge_inventory(endpoint)
        plan = missing_components(remote, bridge)
        save('plan', before=dict(remote=remote, bridge=bridge), plan=plan)
        if check_only:
            save('checked', state='checked', message='只读自检完成；未启动任何组件。')
            return report
        # Starting a missing bridge is read/control-neutral and lets us inspect
        # Teach/SAFE before starting drivers or configuring CAN.
        if 'bridge' in plan:
            if remote['bridge_process']:
                raise DeviceRestartError('bridge 进程存在但无法连接；不重复启动，请检查网络/端口。')
            save('start_bridge')
            start_component(host, workspace, 'bridge')
            report['actions'].append('bridge')
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                bridge = bridge_inventory(endpoint)
                if bridge['online']:
                    break
                time.sleep(1)
            if not bridge['online']:
                raise DeviceRestartError('bridge 启动后仍无法连接；请检查 Dashboard 到 AGX 的网络。')
        assert_idle(bridge, remote)
        if 'can' in plan:
            if remote['arm_nodes']:
                raise DeviceRestartError('驱动仍在运行但 CAN 不健康；不能在线重配置 CAN，请人工检查。')
            save('configure_can')
            _ensure_can(host, workspace, sudo_password_file)
            report['actions'].append('can')
        for component in ('arms', 'cameras', 'preview'):
            if component not in plan:
                continue
            if component == 'arms':
                running = remote['arm_nodes'] or remote['arbiter'] or remote['arm_launch']
            elif component == 'cameras':
                running = remote['camera_nodes'] or remote['camera_launch']
            else:
                running = remote['preview_process'] or bool(remote.get('preview'))
            if running:
                # Let already-running drivers settle; the final fresh-data
                # check decides health. Never spawn a competing driver.
                print(f'{component}: 已有进程，保留并等待数据复检。', flush=True)
                continue
            assert_idle(bridge_inventory(endpoint), remote)
            if component == 'cameras':
                _ensure_realsense_cameras(host)
            save(f'start_{component}')
            start_component(host, workspace, component)
            report['actions'].append(component)
        save('verify')
        deadline = time.monotonic() + timeout_s
        stable_since = None
        previous_versions = None
        missing = missing_components(remote, bridge)
        while time.monotonic() < deadline:
            bridge = bridge_inventory(endpoint)
            assert_idle(bridge, {})
            remote = remote_inventory(host, workspace)
            missing = missing_components(remote, bridge)
            versions = [remote.get('preview', {}).get(n, {}).get('version', 0) for n in CAMERAS]
            advancing = previous_versions is not None and all(a > b for a, b in zip(versions, previous_versions))
            previous_versions = versions
            safe = (bridge['control'].get('effective_mode') == 'SAFE'
                    and bridge['control'].get('output_enabled') is False
                    and bridge['control'].get('teach_active') is False)
            if not missing and advancing and safe:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= 3:
                    save('ready', state='ready', after=dict(remote=remote, bridge=bridge),
                         message='四臂反馈、三相机和预览持续更新；SAFE，运动输出关闭。')
                    return report
            else:
                stable_since = None
            time.sleep(1)
        raise DeviceRestartError(f'复检未通过：缺失/异常组件 {missing}；现有异常进程未被强制重启。',
                                 status=dict(remote=remote, bridge=bridge))
    except Exception as exc:
        save('failed', state='failed', error=str(exc), details=getattr(exc, 'status', {}))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ssh-host', required=True)
    p.add_argument('--remote-workspace', type=Path, required=True)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--sudo-password-file', type=Path)
    p.add_argument('--timeout', type=float, default=90)
    a = p.parse_args()
    ensure_robot(a.ssh_host, a.remote_workspace, a.endpoint, a.report,
                 check_only=a.check_only, timeout_s=a.timeout, sudo_password_file=a.sudo_password_file)


if __name__ == '__main__':
    main()
