"""Smooth V3 policy runner with robust ensembling and jerk-limited targets."""

from .run_lerobot_policy_smooth import SmoothRunnerProfile, main


V3_PROFILE = SmoothRunnerProfile(
    name="Smooth V3",
    target_filter_tau=0.04,
    consistency_scale=0.12,
    chunk_smoothing_passes=2,
    max_joint_velocity=0.42,
    max_joint_acceleration=0.80,
    max_joint_jerk=4.0,
    max_gripper_velocity=0.07,
    max_gripper_acceleration=0.16,
    max_gripper_jerk=0.8,
)


if __name__ == "__main__":
    main(V3_PROFILE)
