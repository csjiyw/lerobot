from __future__ import annotations

from pathlib import Path

import pytest

import cobot_bridge.init_then_record as init_then_record


class ExecCalled(Exception):
    pass


class CompletedChild:
    returncode = 0

    def poll(self):
        return 0


def test_successful_init_execs_recorder_after_recording_phase(tmp_path, monkeypatch):
    phase_file = tmp_path / "episode.phase"
    stop_file = tmp_path / "episode.result"
    executed = {}
    monkeypatch.setattr(init_then_record, "STOP", False)
    monkeypatch.setattr(init_then_record.subprocess, "Popen", lambda _command: CompletedChild())

    def fake_exec(executable, command):
        executed["executable"] = executable
        executed["command"] = command
        raise ExecCalled

    monkeypatch.setattr(init_then_record.os, "execv", fake_exec)
    monkeypatch.setattr(
        init_then_record.sys,
        "argv",
        [
            "init_then_record",
            "--endpoint",
            "tcp://robot:5555",
            "--phase-file",
            str(phase_file),
            "--stop-file",
            str(stop_file),
            "--",
            "--output",
            str(tmp_path / "episode.raw.h5"),
        ],
    )

    with pytest.raises(ExecCalled):
        init_then_record.main()

    assert phase_file.read_text() == "recording"
    assert executed["command"][1:3] == ["-m", "cobot_bridge.record_raw"]
    assert "--output" in executed["command"]


def test_abort_during_init_never_starts_recorder(tmp_path, monkeypatch):
    phase_file = tmp_path / "episode.phase"
    stop_file = tmp_path / "episode.result"

    class InitializingChild:
        returncode = None

        def poll(self):
            stop_file.write_text("discard")
            return None

        def send_signal(self, _signal):
            self.returncode = 0

        def wait(self, timeout):
            assert timeout == 8.0
            return self.returncode

    monkeypatch.setattr(init_then_record, "STOP", False)
    monkeypatch.setattr(init_then_record.subprocess, "Popen", lambda _command: InitializingChild())
    monkeypatch.setattr(
        init_then_record.os,
        "execv",
        lambda *_args: pytest.fail("recorder must not start after abort"),
    )
    monkeypatch.setattr(
        init_then_record.sys,
        "argv",
        [
            "init_then_record",
            "--endpoint",
            "tcp://robot:5555",
            "--phase-file",
            str(phase_file),
            "--stop-file",
            str(stop_file),
            "--",
            "--output",
            str(tmp_path / "episode.raw.h5"),
        ],
    )

    init_then_record.main()

    assert phase_file.read_text() == "aborted_before_record"
    assert not stop_file.exists()
    assert not (tmp_path / "episode.raw.h5").exists()
