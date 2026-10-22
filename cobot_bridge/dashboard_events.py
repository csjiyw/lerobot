"""One lightweight runtime observer shared by all SSE dashboard clients."""
from collections import deque
import json
import logging
import threading
import time


LOG = logging.getLogger(__name__)


class DashboardEvents:
    def __init__(self, jobs, policy_status, rlt_status, read_raw, *, capacity=128, robot_status=None):
        self.jobs = jobs
        self.policy_status = policy_status
        self.rlt_status = rlt_status
        self.read_raw = read_raw
        self.robot_status = robot_status
        self.robot_revision = 0
        self.robot_checked_at = None
        self.previous_robot = None
        self.lock = threading.Lock()
        self.events = deque(maxlen=capacity)
        self.sequence = 0
        self.runtime = {'jobs': [], 'policy_server': None, 'rlt_policy_server': None, 'raws': []}
        self.previous_jobs = None
        self.previous_servers = {}
        self.stop_event = threading.Event()
        self.thread = None
        self.worker_threads = []

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, daemon=True, name='dashboard-events')
            self.thread.start()
            for name, target in [('dashboard-model-events', self._run_servers),
                                 ('dashboard-robot-status', self._run_robot)]:
                if name == 'dashboard-robot-status' and self.robot_status is None:
                    continue
                worker = threading.Thread(target=target, daemon=True, name=name)
                self.worker_threads.append(worker)
                worker.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        for worker in self.worker_threads:
            worker.join(timeout=3)

    def observe_robot(self):
        # Dedicated worker. No file reads, model probes, or queue backlog on
        # this path; only compact discrete status, never joint/image payloads.
        status = self.robot_status()
        fingerprint = self.fingerprint(status)
        with self.lock:
            if fingerprint != self.previous_robot:
                self.robot_revision += 1
                self.previous_robot = fingerprint
                self.runtime['robot_status'] = status
            self.robot_checked_at = time.monotonic()

    def robot_event_after(self, revision):
        with self.lock:
            status = self.runtime.get('robot_status')
            if status is None:
                return None
            age = max(0., (time.monotonic() - self.robot_checked_at) * 1000)
            changed = revision != self.robot_revision
            return {'event': 'robot_status_changed' if changed else 'robot_heartbeat',
                    'data': {'revision': self.robot_revision, 'age_ms': round(age, 1),
                             **({'status': status} if changed else {})}}

    def _run_robot(self):
        while not self.stop_event.is_set():
            try:
                self.observe_robot()
            except Exception:
                LOG.exception('robot status observer failed')
                # Do not advance checked_at: client marks the previous status stale.
            self.stop_event.wait(0.2)

    @staticmethod
    def fingerprint(value):
        # A PING timestamp is not a state transition.
        if isinstance(value, dict):
            value = {k: v for k, v in value.items() if k != 'server_time_ns'}
        return json.dumps(value, sort_keys=True, default=str)

    def publish(self, kind, data):
        with self.lock:
            self.sequence += 1
            self.events.append({'id': self.sequence, 'event': kind, 'data': data})

    def snapshot(self):
        with self.lock:
            return {'id': self.sequence, 'event': 'runtime_snapshot', 'data': dict(self.runtime)}

    def after(self, cursor):
        with self.lock:
            if cursor > self.sequence or (self.events and cursor < self.events[0]['id'] - 1):
                return None  # Client must resynchronize after overrun/server restart.
            return [event for event in self.events if event['id'] > cursor]

    def observe(self, *, check_servers=True):
        jobs = self.jobs()
        with self.lock:
            self.runtime['jobs'] = jobs
        current = {j['id']: self.fingerprint(j) for j in jobs}
        if self.previous_jobs is not None:
            for job in jobs:
                if current[job['id']] == self.previous_jobs.get(job['id']):
                    continue
                data = {'job': job, 'jobs': jobs}
                if job['kind'] in ('record', 'record_with_init') and job['state'] in ('completed', 'failed'):
                    output = job.get('metadata', {}).get('output')
                    if output:
                        try:
                            data['raw'] = self.read_raw(output)
                            if data['raw']:
                                with self.lock:
                                    recent = {raw['id']: raw for raw in self.runtime['raws']}
                                    recent[data['raw']['id']] = data['raw']
                                    self.runtime['raws'] = list(recent.values())[-128:]
                        except (OSError, ValueError):
                            # Aborted episodes may deliberately remove their raw file.
                            data['raw'] = None
                self.publish('job_changed', data)
        self.previous_jobs = current
        if check_servers:
            self.observe_servers()

    def observe_servers(self):
        for name, callback in [('policy_server', self.policy_status), ('rlt_policy_server', self.rlt_status)]:
            status = callback()
            with self.lock:
                self.runtime[name] = status
            fingerprint = self.fingerprint(status)
            if fingerprint != self.previous_servers.get(name):
                self.publish('server_changed', {'name': name, 'status': status})
            self.previous_servers[name] = fingerprint

    def _run_servers(self):
        while not self.stop_event.is_set():
            try:
                self.observe_servers()
            except Exception:
                LOG.exception('model status observer failed')
            self.stop_event.wait(2.0)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.observe(check_servers=False)
            except Exception:
                LOG.exception('dashboard runtime event observer failed')
            self.stop_event.wait(0.5)
