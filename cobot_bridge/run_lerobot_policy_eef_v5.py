"""EEF V5 rollout: standard policy loop with Cartesian smoothing before Pink IK."""

from __future__ import annotations

import sys

from .run_lerobot_policy import main


if __name__ == "__main__":
    # Keep the standard runner as the single owner of policy/Teach/SAFE logic.
    if "--eef-cartesian-smooth" not in sys.argv:
        sys.argv.append("--eef-cartesian-smooth")
    main()
