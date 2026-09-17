"""Move all four arms to init, then start a raw IL recorder.

The recorder is executed only after ``move_init_pose`` exits successfully, so
the alignment trajectory is never part of the IL episode.  A result file
created during initialization aborts the sequence before recording starts.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .move_init_pose import DEFAULT_JOINT_SPEED, DEFAULT_MIN_SECONDS


LOG = logging.getLogger("cobot_bridge.init_then_record")
STOP = False


def _stop(_signal: int, _frame: Any) -> None:
    global STOP
    STOP = True


def _write_phase(path: Path, phase: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(phase)
    os.replace(temporary, path)
    LOG.info("phase=%s", phase)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--phase-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("record_arguments", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    record_arguments = list(args.record_arguments)
    if record_arguments and record_arguments[0] == "--":
        record_arguments.pop(0)
    if not record_arguments:
        raise ValueError("record_raw arguments are required after --")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    if args.stop_file.exists():
        raise FileExistsError(f"result file already exists: {args.stop_file}")

    _write_phase(args.phase_file, "initializing")
    init_command = [
        sys.executable,
        "-m",
        "cobot_bridge.move_init_pose",
        "--endpoint",
        args.endpoint,
        "--joint-speed",
        str(DEFAULT_JOINT_SPEED),
        "--min-seconds",
        str(DEFAULT_MIN_SECONDS),
        "--waypoint-hz",
        "20",
        "--execute",
    ]
    LOG.info("starting four-arm init pose")
    child = subprocess.Popen(init_command)
    while child.poll() is None:
        if STOP or args.stop_file.exists():
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=3.0)
            if args.stop_file.exists():
                args.stop_file.unlink(missing_ok=True)
                _write_phase(args.phase_file, "aborted_before_record")
                LOG.info("init-and-record aborted before recording")
                return
            _write_phase(args.phase_file, "interrupted")
            raise KeyboardInterrupt
        time.sleep(0.1)

    if child.returncode != 0:
        _write_phase(args.phase_file, "init_failed")
        raise subprocess.CalledProcessError(child.returncode, init_command)
    if STOP or args.stop_file.exists():
        args.stop_file.unlink(missing_ok=True)
        _write_phase(args.phase_file, "aborted_before_record")
        return

    _write_phase(args.phase_file, "recording")
    record_command = [sys.executable, "-m", "cobot_bridge.record_raw", *record_arguments]
    LOG.info("init complete; starting raw IL recorder")
    os.execv(sys.executable, record_command)


if __name__ == "__main__":
    main()
