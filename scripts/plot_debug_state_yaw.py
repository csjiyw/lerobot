"""Plot measured state yaw from raw joint feedback (offline; no robot I/O)."""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from cobot_bridge.piper_eef_metrics import PiperForwardKinematics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with h5py.File(args.raw) as raw:
        joints = raw['frames/puppet_position'][:, :6]
        stamps = raw['frames/pc_timestamp_ns'][:]
        job_id = str(raw.attrs.get('debug_job_id', ''))
    times = (stamps - stamps[0]) / 1e9
    fk = PiperForwardKinematics(frames=('gripper_center',))
    _, matrices = fk.forward(joints)['gripper_center']
    yaw = np.rad2deg(np.unwrap(Rotation.from_matrix(matrices).as_euler('xyz')[:, 2]))
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'],
                         'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    # The annotated events belong only to this inspected success rollout.
    events = [(235, '抓住工件'), (595, '松爪前'), (607, '已松爪'), (707, '再次靠近下压')]
    for ax in axes:
        ax.plot(times, yaw, color='#2278b5', linewidth=1.6, label='实测 state yaw（未平滑）')
        ax.set_ylabel('Yaw（°）')
        ax.set_xlabel('距首个记录帧的时间（秒）')
        ax.grid(alpha=.25)
        ax.axhline(0, color='gray', linewidth=.7, alpha=.6)
    axes[0].set_title(f'成功轨迹 · 实测 state yaw · {job_id}')
    axes[0].set_xlim(0, times[-1])
    axes[0].legend(loc='upper right')
    axes[1].set_title('抓取—放置—后续下压段放大')
    axes[1].set_xlim(12, 53)
    if job_id == 'debug_091a720d6a1b':
        for frame, label in events:
            for ax in axes:
                ax.axvline(times[frame], color='#d68a31', linestyle='--', alpha=.5, linewidth=.8)
            dx, dy = {'抓住工件':(-15,28), '松爪前':(-95,-40),
                      '已松爪':(22,26), '再次靠近下压':(-100,-12)}[label]
            axes[1].scatter(times[frame], yaw[frame], s=28, color='#c16a16', zorder=4)
            axes[1].annotate(f'{label}\nf{frame} · {yaw[frame]:.1f}°',
                             xy=(times[frame], yaw[frame]), xytext=(dx,dy),
                             textcoords='offset points', fontsize=9,
                             arrowprops=dict(arrowstyle='-', color='#777777'))
    fig.supxlabel('来源：puppet_position 实测关节 → gripper_center FK → XYZ 欧拉角 yaw；不是模型目标角度', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    np.savetxt(args.output.with_suffix('.csv'), np.column_stack((np.arange(len(times)), times, yaw)),
               delimiter=',', header='frame,time_s,state_yaw_deg', comments='', fmt=['%d','%.6f','%.6f'])
    print(args.output.resolve())



if __name__ == '__main__':
    main()
