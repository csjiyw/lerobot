from copy import deepcopy

from cobot_bridge.dashboard_events import DashboardEvents


def make_hub(capacity=128):
    state = {'jobs': [], 'server': {'ready': False, 'server_time_ns': 1}}
    reads = []
    def read_raw(path):
        reads.append(path)
        return {'id': path, 'completed': True, 'outcome': 'success'}
    hub = DashboardEvents(lambda: deepcopy(state['jobs']), lambda: dict(state['server']),
                          lambda: {'ready': False}, read_raw, capacity=capacity)
    return hub, state, reads


def test_record_completion_reads_only_finished_file_once():
    hub, state, reads = make_hub()
    state['jobs'] = [{'id':'rec','kind':'record','state':'running','metadata':{'output':'one.raw.h5'}}]
    hub.observe()
    baseline = hub.sequence
    state['jobs'][0]['state'] = 'completed'
    hub.observe(check_servers=False)
    events = hub.after(baseline)
    assert len(events) == 1 and events[0]['event'] == 'job_changed'
    assert events[0]['data']['raw']['outcome'] == 'success'
    assert reads == ['one.raw.h5']
    hub.observe(check_servers=False)
    assert len(hub.after(baseline)) == 1 and reads == ['one.raw.h5']
    assert hub.snapshot()['data']['raws'][0]['id'] == 'one.raw.h5'


def test_server_timestamp_is_not_a_notification_and_ready_is():
    hub, state, _ = make_hub()
    hub.observe();cursor = hub.sequence
    state['server']['server_time_ns'] += 1
    hub.observe()
    assert hub.after(cursor) == []
    state['server']['ready'] = True
    hub.observe()
    assert hub.after(cursor)[0]['data']['status']['ready'] is True


def test_replay_overrun_restart_and_initial_history():
    hub, state, reads = make_hub(capacity=2)
    state['jobs'] = [{'id':'old','kind':'record','state':'completed','metadata':{'output':'old.raw.h5'}}]
    hub.observe(check_servers=False)
    assert reads == []  # No false "saved" notifications for historical jobs.
    for i in range(3): hub.publish('job_changed', {'i': i})
    assert hub.after(0) is None
    assert [event['id'] for event in hub.after(1)] == [2,3]
    assert hub.after(99) is None
    assert hub.snapshot()['id'] == 3


def test_robot_latest_status_is_separate_from_task_backlog():
    hub, state, reads = make_hub(capacity=2)
    robot = {'online': True, 'mode': 'SAFE', 'teach_active': False}
    hub.robot_status = lambda: dict(robot)
    hub.observe_robot()
    first = hub.robot_event_after(None)
    assert first['event'] == 'robot_status_changed'
    assert first['data']['revision'] == 1
    assert 'id' not in first
    assert hub.robot_event_after(1)['event'] == 'robot_heartbeat'
    hub.observe_robot()
    assert hub.robot_revision == 1  # Stable status produces no transition.
    for i in range(5):
        hub.publish('job_changed', {'i': i})
    robot['teach_active'] = True
    hub.observe_robot()
    assert hub.robot_event_after(1)['data']['status']['teach_active'] is True
    assert hub.robot_event_after(1)['data']['revision'] == 2
    assert reads == []


def test_slow_model_probe_does_not_block_robot_worker():
    import threading
    hub, state, _ = make_hub()
    blocked = threading.Event()
    release = threading.Event()
    def slow():
        blocked.set()
        release.wait(3)
        return {'ready': False}
    hub.policy_status = slow
    hub.robot_status = lambda: {'online': True, 'mode': 'TEACH'}
    worker = threading.Thread(target=hub.observe_servers)
    worker.start()
    try:
        assert blocked.wait(1)
        hub.observe_robot()
        assert hub.robot_event_after(None)['data']['status']['mode'] == 'TEACH'
        assert worker.is_alive()
    finally:
        release.set()
        worker.join(2)


def test_robot_observation_age_exposes_stalled_detection(monkeypatch):
    import cobot_bridge.dashboard_events as mod
    clock = [1.]
    monkeypatch.setattr(mod.time, 'monotonic', lambda: clock[0])
    hub, _, _ = make_hub()
    hub.robot_status = lambda: {'online': True}
    hub.observe_robot()
    clock[0] = 4.
    assert hub.robot_event_after(1)['data']['age_ms'] == 3000
