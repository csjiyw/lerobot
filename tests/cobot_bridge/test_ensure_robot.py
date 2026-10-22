import json
from pathlib import Path

import pytest

from cobot_bridge import ensure_robot as mod


def inventory():
    return dict(can={n: True for n in mod.CAN_INTERFACES}, arm_nodes=0,
                arm_launch=False, arbiter=False, camera_nodes=0, camera_launch=False,
                bridge_process=False, preview_process=False, preview={}, external_activity=False)


def bridge(online=True):
    return dict(online=online, arms={f'{r}_{s}': online for r in ('master','puppet') for s in ('left','right')},
                cameras={n: online for n in mod.CAMERAS},
                control=dict(effective_mode='SAFE', output_enabled=False, teach_active=False))


def mock_io(monkeypatch, remote, observed):
    calls=[]
    monkeypatch.setattr(mod, 'remote_inventory', lambda *a: remote)
    monkeypatch.setattr(mod, 'bridge_inventory', lambda *a: observed)
    monkeypatch.setattr(mod, 'start_component', lambda h,w,c: calls.append(c))
    monkeypatch.setattr(mod, '_ensure_can', lambda *a: calls.append('can'))
    monkeypatch.setattr(mod, '_ensure_realsense_cameras', lambda *a: None)
    return calls


def test_check_only_never_changes_services(tmp_path, monkeypatch):
    r=inventory();r['can']={}
    calls=mock_io(monkeypatch,r,bridge(False))
    p=tmp_path/'report.json'
    result=mod.ensure_robot('agx',Path('/remote'),'tcp://agx:5555',p,check_only=True)
    assert calls==[]
    assert result['plan']==['can','bridge','arms','cameras','preview']
    assert json.loads(p.read_text())['state']=='checked'


@pytest.mark.parametrize('field',['teach_active','output_enabled'])
def test_active_control_blocks_start(tmp_path,monkeypatch,field):
    b=bridge();b['control'][field]=True
    calls=mock_io(monkeypatch,inventory(),b)
    with pytest.raises(mod.DeviceRestartError):
        mod.ensure_robot('agx',Path('/remote'),'tcp://agx:5555',tmp_path/'r.json')
    assert calls==[]


def test_missing_preview_only_starts_preview(tmp_path,monkeypatch):
    r=inventory();r['arm_nodes']=4;r['camera_nodes']=3
    calls=mock_io(monkeypatch,r,bridge())
    # timeout zero ends in failure: starting a process alone must not count as ready.
    with pytest.raises(mod.DeviceRestartError):
        mod.ensure_robot('agx',Path('/remote'),'tcp://agx:5555',tmp_path/'r.json',timeout_s=0)
    assert calls==['preview']


def test_existing_unhealthy_drivers_not_duplicated(tmp_path,monkeypatch):
    r=inventory();r.update(arm_nodes=2,arbiter=True,camera_nodes=1,preview_process=True)
    b=bridge();b['arms']={};b['cameras']={}
    calls=mock_io(monkeypatch,r,b)
    with pytest.raises(mod.DeviceRestartError):
        mod.ensure_robot('agx',Path('/remote'),'tcp://agx:5555',tmp_path/'r.json',timeout_s=0)
    assert calls==[]


def test_can_not_reconfigured_with_live_drivers(tmp_path,monkeypatch):
    r=inventory();r.update(arm_nodes=4,can={})
    calls=mock_io(monkeypatch,r,bridge())
    with pytest.raises(mod.DeviceRestartError,match='CAN'):
        mod.ensure_robot('agx',Path('/remote'),'tcp://agx:5555',tmp_path/'r.json')
    assert calls==[]


def test_ready_requires_advancing_preview_and_safe_state(tmp_path,monkeypatch):
    r=inventory();r.update(arm_nodes=4,camera_nodes=3)
    r['preview']={n:dict(version=1) for n in mod.CAMERAS}
    calls=mock_io(monkeypatch,r,bridge())
    clock=[0.]
    monkeypatch.setattr(mod.time,'monotonic',lambda:clock[0])
    def sleep(seconds):
        clock[0]+=seconds
        for v in r['preview'].values():v['version']+=1
    monkeypatch.setattr(mod.time,'sleep',sleep)
    result=mod.ensure_robot('agx',Path('/remote'),'tcp://agx:5555',tmp_path/'r.json',timeout_s=10)
    assert result['state']=='ready'
    assert calls==[]


def test_stalled_preview_cannot_pass(tmp_path,monkeypatch):
    r=inventory();r.update(arm_nodes=4,camera_nodes=3,preview_process=True)
    r['preview']={n:dict(version=1) for n in mod.CAMERAS}
    calls=mock_io(monkeypatch,r,bridge())
    clock=[0.]
    monkeypatch.setattr(mod.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(mod.time,'sleep',lambda s:clock.__setitem__(0,clock[0]+s))
    with pytest.raises(mod.DeviceRestartError):
        mod.ensure_robot('agx',Path('/remote'),'tcp://agx:5555',tmp_path/'r.json',timeout_s=3)
    assert calls==[]
