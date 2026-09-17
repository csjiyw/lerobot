"""PC-side bridge smoke test. Read-only unless ``--exercise-control`` is set."""

from __future__ import annotations

import argparse
import logging
import statistics
import time

from .client import CobotBridgeClient


LOG = logging.getLogger("cobot_bridge.smoke_test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://192.168.250.2:5555")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--images",
        nargs="*",
        choices=("front", "left", "right"),
        default=["front", "left", "right"],
    )
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument(
        "--exercise-control",
        action="store_true",
        help="exercise mode/epoch/action through a process-level dry-run arbiter",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    latencies: list[float] = []
    ready_count = 0
    last_observation = None

    with CobotBridgeClient(args.endpoint, timeout_s=args.timeout) as client:
        LOG.info("ping=%s", client.ping())
        period = 1.0 / max(1.0, args.fps)
        for index in range(args.cycles):
            started = time.perf_counter()
            observation, images = client.get_observation(
                sides=(args.side,), images=tuple(args.images)
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)
            ready_count += int(bool(observation.get("ready")))
            last_observation = observation
            if index in (0, args.cycles - 1):
                LOG.info(
                    "cycle=%d ready=%s missing=%s sequence=%s image_bytes=%s latency_ms=%.2f",
                    index + 1,
                    observation.get("ready"),
                    observation.get("missing_or_stale"),
                    observation.get("sequence"),
                    {name: len(data) for name, data in images.items()},
                    latency_ms,
                )
            time.sleep(max(0.0, period - (time.perf_counter() - started)))

        LOG.info(
            "observation_test cycles=%d ready=%d latency_ms(mean/p95)=%.2f/%.2f",
            args.cycles,
            ready_count,
            statistics.mean(latencies),
            sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        )
        if args.require_ready and ready_count != args.cycles:
            raise RuntimeError("one or more observations were not ready")

        if args.exercise_control:
            if last_observation is None:
                raise RuntimeError("no observation available")
            control = last_observation.get("control", {})
            if not control.get("dry_run", False):
                raise RuntimeError("refusing control exercise: ROS arbiter is not dry-run")
            try:
                LOG.info("set_mode=%s", client.set_mode(2))
                LOG.info("set_output=%s", client.set_output(True))
                current, _ = client.get_observation(sides=(args.side,), images=())
                epoch = int(current.get("control", {}).get("epoch", -1))
                state = current.get("puppet", {}).get(args.side)
                if epoch < 0 or not isinstance(state, dict):
                    raise RuntimeError("missing epoch or puppet state")
                result = client.set_action(
                    args.side,
                    state["position"],
                    epoch=epoch,
                    sequence=1,
                )
                LOG.info("dry_run_action_result=%s", result)
                if result.get("accepted") or result.get("reason") != "dry_run":
                    raise RuntimeError(f"expected dry_run rejection, got {result}")
                resync_result = client.resync()
                LOG.info("resync=%s", resync_result)
                after_resync, _ = client.get_observation(sides=(args.side,), images=())
                new_epoch = int(after_resync.get("control", {}).get("epoch", -1))
                if new_epoch <= epoch:
                    raise RuntimeError(f"epoch did not advance after resync: {epoch} -> {new_epoch}")
                stale = client.set_action(
                    args.side,
                    state["position"],
                    epoch=epoch,
                    sequence=2,
                )
                LOG.info("stale_epoch_action_result=%s", stale)
                if stale.get("reason") != "stale_epoch":
                    raise RuntimeError(f"expected stale_epoch rejection, got {stale}")
                fresh = client.set_action(
                    args.side,
                    state["position"],
                    epoch=new_epoch,
                    sequence=3,
                )
                LOG.info("new_epoch_dry_run_result=%s", fresh)
                if fresh.get("reason") != "dry_run":
                    raise RuntimeError(f"expected dry_run rejection, got {fresh}")
            finally:
                LOG.info("disable_output=%s", client.set_output(False))
                LOG.info("safe_mode=%s", client.set_mode(0))


if __name__ == "__main__":
    main()
