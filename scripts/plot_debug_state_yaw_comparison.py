"""Compare dense success feedback against retained sparse failure decision states."""
import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from cobot_bridge.piper_eef_metrics import PiperForwardKinematics


def trace(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    states = np.array([r['model_state'] for r in rows])
    stamps = np.array([r['bridge_time_ns'] for r in rows], dtype=np.int64)
    t = (stamps - stamps[0]) / 1e9
    yaw = np.rad2deg(np.unwrap(Rotation.from_rotvec(states[:, 3:6]).as_euler('xyz')[:, 2]))
    # Comparable observed event, NOT the exact physical opening onset.
    matches = np.flatnonzero((np.arange(len(rows)) >= 10) & (states[:, 6] > .06)
                            & (states[:, 0] < .4) & (states[:, 2] < .15))
    if not len(matches):
        raise ValueError(f'No placement opening observation in {path}')
    return t, yaw, int(matches[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--success-raw', type=Path, required=True)
    p.add_argument('--success-trace', type=Path, required=True)
    p.add_argument('--failure-trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    ts, ys, os = trace(a.success_trace)
    tf, yf, of = trace(a.failure_trace)
    with h5py.File(a.success_raw) as raw:
        joints = raw['frames/puppet_position'][:, :6]
        stamps = raw['frames/pc_timestamp_ns'][:]
        indices = raw['policy_decisions/raw_frame_index'][:]
    fk = PiperForwardKinematics(frames=('gripper_center',))
    _, mat = fk.forward(joints)['gripper_center']
    yd = np.rad2deg(np.unwrap(Rotation.from_matrix(mat).as_euler('xyz')[:, 2]))
    td = (stamps - stamps[int(indices[0])]) / 1e9
    open_dense = float((stamps[int(indices[os])] - stamps[int(indices[0])]) / 1e9)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK JP', 'DejaVu Sans'], 'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    for k, ax in enumerate(axes):
        sd, ss, ff = (0, 0, 0) if k == 0 else (open_dense, ts[os], tf[of])
        ax.plot(td-sd, yd, color='#2278b5', lw=1.3, label='成功 10/20：逐帧实测 state')
        ax.plot(ts-ss, ys, 'o', color='#2278b5', ms=3, alpha=.6, label='成功：推理输入 state 采样点')
        ax.plot(tf-ff, yf, 'o--', color='#d95f02', ms=4, lw=1.5,
                label='失败 10/15：仅保留推理输入 state（约每2秒；虚线非逐帧）')
        ax.grid(alpha=.25)
        ax.set_ylabel('State yaw（°）')
    axes[0].set_xlim(0, 55)
    axes[0].set_title('成功 / 失败 yaw 对比：从首次推理观测起计时')
    axes[0].set_xlabel('时间（秒）')
    axes[0].legend(fontsize=9, loc='upper left')
    axes[1].set_xlim(-18, 12)
    axes[1].axvline(0, color='gray', linestyle=':', lw=1.2)
    axes[1].set_title('放置末段对齐：0秒 = 首次推理观测到放置区夹爪开度 >60mm（不是精确开爪瞬间）')
    axes[1].set_xlabel('相对上述开爪观测的时间（秒）')
    axes[1].annotate(f'成功 {ys[os]:.1f}° / 失败 {yf[of]:.1f}°',
                     xy=(0, ys[os]), xytext=(2, 17),
                     arrowprops=dict(arrowstyle='->', color='gray'))
    fig.supxlabel('两者均为实测 state 的 XYZ 欧拉角 yaw，非模型目标；失败记录已滚动覆盖，不能恢复逐帧细节。', fontsize=10)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output, dpi=160)
    plt.close(fig)
    csv = np.concatenate([np.column_stack((np.ones(len(ts)), ts, ts-ts[os], ys)),
                          np.column_stack((np.zeros(len(tf)), tf, tf-tf[of], yf))])
    np.savetxt(a.output.with_suffix('.csv'), csv, delimiter=',',
               header='success_1_failure_0,time_from_first_decision_s,time_from_open_observation_s,state_yaw_deg', comments='')
    print(json.dumps({'output': str(a.output), 'success_open_yaw': ys[os],
                      'failure_open_yaw': yf[of], 'success_points': len(ts), 'failure_points': len(tf)}))


if __name__ == '__main__':
    main()
