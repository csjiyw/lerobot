import h5py
import numpy as np
import matplotlib.pyplot as plt

h5_path = "data/rclone-v1.68.2-linux-amd64/data/9f22b6b23ed64ced8fc8074d454e5672/s1a0a654702c4a5482b6895b3c1ad480/s1a0a654702c4a5482b6895b3c1ad480.h5"

k_action = "joints/action/arm/position"
k_state  = "joints/state/arm/position"

with h5py.File(h5_path, "r") as f:
    action = np.array(f[k_action])   # (T, 12)
    state  = np.array(f[k_state])    # (T, 12)

assert action.shape == state.shape, (action.shape, state.shape)
T, J = action.shape
t = np.arange(T)

# 可选：如果你知道控制频率，比如 30Hz，可以换成秒轴
# fps = 30
# t = t / fps

out_dir = "./joint_plots"
import os
os.makedirs(out_dir, exist_ok=True)

for j in range(J):
    plt.figure()
    plt.plot(t, action[:, j], label="action/arm/position")
    plt.plot(t, state[:, j],  label="state/arm/position")
    plt.title(f"Joint {j}: action vs state")
    plt.xlabel("t (frame index)")
    plt.ylabel("position")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"joint_{j:02d}.png"), dpi=300)
    plt.close()

print(f"Saved {J} plots to: {out_dir}/")
