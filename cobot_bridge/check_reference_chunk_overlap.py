"""Offline native-30Hz overlap diagnostic; no robot communication."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from .run_lerobot_policy import _load_policy
from .policy_server import _restore_relative_eef_chunk
from .test_rlt_open_loop_recording import differences


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', nargs='+', type=int, default=[0, 20, 40])
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    ds = LeRobotDataset('local/cobot_magic_il_2_trimmed_unified_metal_fixture_v2_drop_ep0_3_eef',
                         root=args.dataset, video_backend='pyav', image_transforms=None,
                         delta_timestamps={'action': [i/30 for i in range(30)]})
    assert ds.fps == 30
    policy, pre, post = _load_policy(argparse.Namespace(policy_path=str(args.policy.resolve()),
                                     device=args.device, vlm_model_path=None, n_action_steps=None))
    assert policy.config.chunk_size == 30
    modes = ['fresh', 'fixed_slots', 'time_aligned']
    summary = {'motion': False, 'fps': 30, 'advance_frames': 10,
               'policy': str(args.policy.resolve()), 'dtype': 'float16',
               'limitation': 'Teacher-forced successful IL observations, not closed-loop success evaluation.',
               'episodes': {}}
    saved = {}
    for ep in args.episodes:
        meta = ds.meta.episodes[ep]
        start, stop = int(meta['dataset_from_index']), int(meta['dataset_to_index'])
        indices = np.arange(start, stop-29, 10)
        rng = np.random.default_rng(2000+ep)
        noise_timeline = rng.standard_normal((stop-start+30, policy.config.max_action_dim)).astype(np.float32)
        fixed_noise = rng.standard_normal((1,30,policy.config.max_action_dim)).astype(np.float32)
        outputs = {mode: [] for mode in modes}
        gt, states = [], []
        for j, idx in enumerate(indices):
            sample = ds[int(idx)]
            assert not bool(sample['action_is_pad'].any())
            state = sample['observation.state'].numpy()
            states.append(state)
            gt.append(sample['action'].numpy())
            batch = {key: sample[key].unsqueeze(0).to(args.device)
                     for key in policy.config.input_features}
            batch['task'] = [sample['task']]
            batch['robot_type'] = ['cobot_magic_piper']
            batch = pre(batch)
            for mode in modes:
                noise = (fixed_noise if mode == 'fixed_slots' else
                         noise_timeline[int(idx)-start:int(idx)-start+30][None] if mode == 'time_aligned' else
                         rng.standard_normal(fixed_noise.shape).astype(np.float32))
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    pred = policy.predict_action_chunk(dict(batch), noise=torch.from_numpy(noise.copy()).to(args.device))
                    native = torch.stack([post(pred[:, k]) for k in range(30)], 1).squeeze(0).cpu().numpy()
                    outputs[mode].append(_restore_relative_eef_chunk(pre, state, native))
            if j % 15 == 0:
                print(f'episode={ep} observation={j}/{len(indices)}', flush=True)
        gt = np.asarray(gt)
        assert np.array_equal(gt[:-1, 10:30], gt[1:, :20]), 'GT timeline does not align'
        result = {'observations': len(indices), 'gt_overlap_max_abs': 0.0, 'modes': {}}
        for mode in modes:
            out = np.asarray(outputs[mode])
            result['modes'][mode] = {
                'overlap_old_tail20_vs_new_head20': differences(out[:-1,10:],out[1:,:20]),
                'new_start_vs_old_expected_next': differences(out[1:,0],out[:-1,10]),
                'actual_boundary_step': differences(out[1:,0],out[:-1,9]),
                'old_plan_intrinsic_step': differences(out[:-1,10],out[:-1,9]),
                'gt_overlap_error_old': differences(out[:-1,10:],gt[:-1,10:]),
                'gt_overlap_error_new': differences(out[1:,:20],gt[1:,:20]),
                'gt_full_chunk_error': differences(out,gt),
            }
            saved[f'ep{ep}_{mode}'] = out
        result['same_observation_two_noise_draws'] = differences(outputs['fresh'],outputs['fixed_slots'])
        saved[f'ep{ep}_gt'] = gt
        saved[f'ep{ep}_indices'] = indices
        saved[f'ep{ep}_states'] = np.asarray(states)
        summary['episodes'][str(ep)] = result
        print('EPISODE_DONE',ep,json.dumps(result),flush=True)
    np.savez_compressed(args.output/'predictions.npz', **saved)
    (args.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('DONE', args.output, flush=True)


if __name__ == '__main__':
    main()
