"""Read-only robot diagnostic: replay recorded observations, never send commands.

This is teacher-forced open-loop evaluation, not a physics simulation or an
exact reconstruction of the historical stochastic rollout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .piper_eef_metrics import PiperForwardKinematics, DEFAULT_PIPER_URDF
from .piper_eef_transform import PiperEEFFrameTransform
from .piper_pink_ik import PiperPinkIK, PiperIKSettings
from .policy_server import _predict_full_chunk, _restore_relative_eef_chunk
from .rlt_online import resample_action_chunk
from .rlt_policy_server import RLTPolicyRuntime
from .run_lerobot_policy import _load_policy


def stats(x):
    a = np.asarray(x, dtype=float)
    return dict(mean=float(a.mean()), p50=float(np.median(a)),
                p95=float(np.percentile(a, 95)), maximum=float(a.max()))


def differences(a, b):
    a, b = np.asarray(a), np.asarray(b)
    angle = (Rotation.from_rotvec(a[..., 3:6].reshape(-1, 3)).inv()
             * Rotation.from_rotvec(b[..., 3:6].reshape(-1, 3))).magnitude()
    return dict(position_mm=stats(np.linalg.norm(a[..., :3]-b[..., :3], axis=-1)*1000),
                orientation_deg=stats(np.degrees(angle)),
                gripper_abs=stats(np.abs(a[..., 6]-b[..., 6])),
                xyz_signed_mean_mm=((a[..., :3]-b[..., :3]).reshape(-1, 3).mean(0)*1000).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'stage1', 'actor', 'base', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    with h5py.File(args.raw) as f:
        if int(f.attrs['fps']) != 15:
            raise ValueError('This diagnostic boundary comparison expects a 15 FPS debug recording')
        indices = np.arange(0, len(f['frames/puppet_position']), args.stride)
        joints = f['frames/puppet_position'][:][indices]
        time_ns = f['frames/pc_timestamp_ns'][:][indices]
        task = str(f.attrs['task'])
        images = [{name: cv2.cvtColor(cv2.imdecode(f[f'images/{name}/jpeg'][int(i)],
                    cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                   for name in ('front', 'right')} for i in indices]
        source_attrs = {k: str(v) for k, v in f.attrs.items()}
    fk = PiperForwardKinematics(frames=('gripper_center',))
    xyz, rot = fk.forward(joints[:, :6])['gripper_center']
    states = np.c_[xyz, Rotation.from_matrix(rot).as_rotvec(), joints[:, 6]].astype(np.float32)
    runtime = RLTPolicyRuntime(stage1_checkpoint=args.stage1, stage2_checkpoint=args.actor,
                              device=args.device, cameras=('front', 'right'))
    status = runtime.begin_episode(workflow_mode=1, allow_actor=True,
                                   actor_exploration_std=0, q_select_k=1)
    if not runtime.actor_ready:
        raise RuntimeError('Actor did not load')
    print('RUNTIME', json.dumps(status), flush=True)
    outputs = {key: [] for key in ('actor', 'reference', 'standard_stage1', 'base')}
    latency = []
    for seed in args.seeds:
        actors, refs = [], []
        for j, (state, image) in enumerate(zip(states, images)):
            torch.manual_seed(seed + int(indices[j]))
            result = runtime.predict(state=state, images=image, task=task, robot_type='piper')
            actors.append(result['raw_policy_chunk'])
            refs.append(result['raw_reference_full_chunk'])
            latency.append(result['latency_ms'])
            if j % 20 == 0:
                print(f'RLT seed={seed} frame={j}/{len(states)}', flush=True)
        outputs['actor'].append(actors)
        outputs['reference'].append(refs)
    # Use the normal BC server helper, identical states/RGB/text and RNG seeds.
    for name in ('standard_stage1', 'base'):
        if name == 'base':
            policy, pre, post = _load_policy(argparse.Namespace(policy_path=str(args.base.resolve()),
                device=args.device, vlm_model_path=None, n_action_steps=None))
        else:
            policy, pre, post = runtime.policy, runtime.preprocessor, runtime.postprocessor
        for seed in args.seeds:
            chunks = []
            for j, (state, image) in enumerate(zip(states, images)):
                obs = {'observation.state': state,
                       'observation.images.camera_f': image['front'],
                       'observation.images.camera_r': image['right']}
                torch.manual_seed(seed + int(indices[j]))
                chunk = _predict_full_chunk(obs, policy=policy, device=torch.device(args.device),
                    preprocessor=pre, postprocessor=post, task=task, robot_type='piper')
                chunks.append(_restore_relative_eef_chunk(pre, state, chunk))
                if j % 20 == 0:
                    print(f'{name} seed={seed} frame={j}/{len(states)}', flush=True)
            outputs[name].append(chunks)
    outputs = {key: np.asarray(value, dtype=np.float32) for key, value in outputs.items()}
    np.savez_compressed(args.output/'predictions.npz', **outputs, indices=indices, states=states,
                        time_ns=time_ns, joints=joints, seeds=args.seeds)
    h = outputs['actor'].shape[2]
    report = {'motion': False, 'task': task, 'source': str(args.raw.resolve()),
              'source_attrs': source_attrs, 'frames': len(indices), 'seeds': args.seeds,
              'actor': str(args.actor.resolve()), 'stage1': str(args.stage1.resolve()),
              'base': str(args.base.resolve()), 'horizon': h,
              'anchor_source': 'recorded joint FK gripper_center, NOT historical SDK EEF',
              'limitations': ['Recorded failed-policy observations are NOT expert targets.',
                             'No environment propagation; cannot estimate success or causal horizon effects.',
                             'Seed pairing does not imply identical noise if internal RNG draw order differs.'],
              'inference_latency_ms': stats(latency),
              'actor_minus_reference_first10': differences(outputs['actor'], outputs['reference'][:, :, :h]),
              'rlt_reference_vs_standard_same_stage1': differences(outputs['reference'], outputs['standard_stage1']),
              'stage1_reference_vs_base': differences(outputs['reference'], outputs['base'])}
    report['segments'] = {}
    seconds = (time_ns-time_ns[0])*1e-9
    for name, mask in [('before_takeover', seconds < 17), ('after_takeover', seconds >= 17)]:
        if mask.any():
            report['segments'][name] = differences(outputs['actor'][:, mask], outputs['reference'][:, mask, :h])
    report['seed_variation'] = {name: differences(value[0], value[1])
                              for name, value in outputs.items() if len(value) > 1}
    # Isolated chunks, identical initial joints. Match production frame transform,
    # speed=0.5, 7 Pink ticks per wire point, v=0.6, no extra edge filter.
    transform = PiperEEFFrameTransform()
    ik = PiperPinkIK(DEFAULT_PIPER_URDF, settings=PiperIKSettings(max_joint_velocity=0.60))
    ik_records = {}
    for name in ('reference', 'actor', 'base'):
        errors, angles, generated, joint_moves = [], [], [], []
        for j, chunk in enumerate(outputs[name][0, :, :h]):
            ik.reset(ik.coerce_measured_joints(joints[j, :6]))
            ik.set_posture_reference(ik.configuration.q.copy())
            planned = []
            for target in resample_action_chunk(chunk, 0.5):
                link6 = transform.transform(target, source_frame='gripper_center', target_frame='link6')
                pose = ik.pin.SE3(ik.pin.exp3(link6[3:6].astype(float)), link6[:3].astype(float))
                for _ in range(7):
                    step = ik.step(pose)
                errors.append(step.diagnostics.position_error_m*1000)
                angles.append(np.degrees(step.diagnostics.orientation_error_rad))
                planned.append(step.joints.copy())
            generated.append(planned)
            joint_moves.append(np.max(np.abs(np.asarray(planned)-joints[j, :6])))
            if j % 20 == 0:
                print(f'IK {name} frame={j}/{len(states)}', flush=True)
        ik_records[name] = np.asarray(generated)
        report.setdefault('isolated_chunk_ik', {})[name] = {
            'link6_tracking_position_mm': stats(errors), 'link6_tracking_rotation_deg': stats(angles),
            'max_joint_excursion_per_chunk_rad': stats(joint_moves)}
    np.savez_compressed(args.output/'ik_targets.npz', **ik_records, indices=indices)
    report['ik_actor_vs_reference_joint_difference_rad'] = stats(np.abs(ik_records['actor']-ik_records['reference']))
    # Observed-time comparison only: not a generated closed-loop trajectory.
    # At half-speed a 10/30-action plan lasts 2/3 or 2 seconds.
    for name, horizon in [('reference10', h), ('reference30', 30), ('actor10', h)]:
        source = outputs['actor'] if name == 'actor10' else outputs['reference']
        offset = round(horizon / args.stride)
        if offset > 0 and offset < len(indices):
            report.setdefault('recorded_obs_boundary_gap', {})[name] = differences(
                source[:, :-offset, horizon-1], source[:, offset:, 0])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
