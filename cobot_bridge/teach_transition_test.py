"""Verify RLT_SYNC -> TEACH -> RESYNC_REQUIRED -> RLT_SYNC through the bridge.

This diagnostic refuses to run unless the AGX ROS arbiter reports process-level
dry-run, so it cannot publish physical arm commands.
"""

from __future__ import annotations

import argparse
import logging
import time

from .client import CobotBridgeClient


LOG = logging.getLogger("cobot_bridge.teach_transition_test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://192.168.250.2:5555")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with CobotBridgeClient(args.endpoint, timeout_s=3.0) as client:
        initial, _ = client.get_observation(sides=(args.side,), images=())
        if not initial.get("control", {}).get("dry_run", False):
            raise RuntimeError("refusing transition test: ROS arbiter is not dry-run")
        transitions: list[int] = []
        try:
            LOG.info("set_mode=%s", client.set_mode(2))
            LOG.info("set_output=%s", client.set_output(True))
            deadline = time.monotonic() + args.timeout
            resynced = False
            while time.monotonic() < deadline:
                observation, _ = client.get_observation(sides=(args.side,), images=())
                control = observation.get("control", {})
                mode = int(control.get("effective_mode_id", -1))
                if not transitions or transitions[-1] != mode:
                    transitions.append(mode)
                    LOG.info(
                        "mode=%d epoch=%s teach=%s owner=%s",
                        mode,
                        control.get("epoch"),
                        control.get("teach_active"),
                        control.get("owner"),
                    )
                if mode == 4 and not resynced:
                    LOG.info("resync=%s", client.resync())
                    resynced = True
                if resynced and transitions[-4:] == [2, 3, 4, 2]:
                    break
                time.sleep(0.05)
            if transitions[-4:] != [2, 3, 4, 2]:
                raise RuntimeError(f"expected [2, 3, 4, 2], got {transitions}")
            LOG.info("teach transition passed: %s", transitions[-4:])
        finally:
            LOG.info("disable_output=%s", client.set_output(False))
            LOG.info("safe_mode=%s", client.set_mode(0))


if __name__ == "__main__":
    main()
