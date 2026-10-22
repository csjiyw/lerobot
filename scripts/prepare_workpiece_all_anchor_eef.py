"""Build a non-destructive 407-episode EEF training view and chunk30 statistics."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from scipy.spatial.transform import Rotation

from cobot_bridge.convert_lerobot_joint_to_eef import _joint_to_eef
from cobot_bridge.piper_eef_metrics import PiperForwardKinematics
from cobot_bridge.repair_state_derived_gripper import statistics
from lerobot.datasets.utils import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH, flatten_dict, write_episodes, write_info, write_stats, write_tasks
from lerobot.policies.smolvla.relative_eef import chunk_to_relative, relative_chunk_to_absolute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    task = '把零件摆放到电视后盖板对应位置'
    write_tasks(pd.DataFrame({'task_index': [0]}, index=[task]), output)
    fk = PiperForwardKinematics(frames=('gripper_center',))
    rows, provenance, relative_rows = [], [], []
    all_values = {'observation.state': [], 'action': []}
    features = None
    offset = 0
    for source in args.sources:
        source = source.resolve()
        info = json.loads((source/'meta/info.json').read_text())
        assert info['fps'] == 30 and info['codebase_version'] == 'v3.0'
        essential = ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'observation.state', 'action']
        cameras = ['observation.images.camera_f', 'observation.images.camera_r']
        if features is None:
            features = {k: copy.deepcopy(info['features'][k]) for k in essential+cameras}
            for k in all_values:
                features[k]['names'] = ['x','y','z','rx','ry','rz','gripper']
        episodes = pd.concat([pq.read_table(p).to_pandas() for p in sorted((source/'meta/episodes').rglob('*.parquet'))]).sort_values('episode_index')
        cached_path, cached_table = None, None
        for _, episode in episodes.iterrows():
            old_ep = int(episode['episode_index']); new_ep = len(rows)
            path = source / info['data_path'].format(chunk_index=int(episode['data/chunk_index']), file_index=int(episode['data/file_index']))
            if path != cached_path:
                cached_path, cached_table = path, pq.read_table(path, columns=essential)
            selected = cached_table.filter(pa.compute.equal(cached_table['episode_index'], old_ep))
            n = len(selected)
            assert n == int(episode['length'])
            assert np.array_equal(selected['frame_index'].to_numpy(), np.arange(n))
            data = {k: selected[k] for k in essential}
            data.update(episode_index=pa.array(np.full(n,new_ep,dtype=np.int64)), index=pa.array(np.arange(offset,offset+n,dtype=np.int64)), task_index=pa.array(np.zeros(n,dtype=np.int64)))
            values = {}
            for key in all_values:
                joints = np.asarray(selected[key].to_pylist(), dtype=np.float32)
                values[key] = _joint_to_eef(joints, np.ones(n,dtype=bool), fk)
                np.testing.assert_array_equal(values[key][:,6], joints[:,6])
                data[key] = pa.array(values[key].tolist(), type=pa.list_(pa.float32(),7))
                all_values[key].append(values[key])
            dest = output / DEFAULT_DATA_PATH.format(chunk_index=0,file_index=new_ep)
            dest.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(data), dest, compression='snappy')
            row = dict(episode_index=new_ep,length=n,tasks=[task],dataset_from_index=offset,dataset_to_index=offset+n)
            row.update({'data/chunk_index':0,'data/file_index':new_ep,'meta/episodes/chunk_index':0,'meta/episodes/file_index':0})
            row.update(flatten_dict({'stats':{k:statistics(v) for k,v in values.items()}}))
            for camera in cameras:
                video = source / info['video_path'].format(video_key=camera,chunk_index=int(episode[f'videos/{camera}/chunk_index']),file_index=int(episode[f'videos/{camera}/file_index']))
                if not video.is_file(): raise FileNotFoundError(video)
                target = output / DEFAULT_VIDEO_PATH.format(video_key=camera,chunk_index=0,file_index=new_ep)
                target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(video.resolve())
                row.update({f'videos/{camera}/chunk_index':0,f'videos/{camera}/file_index':new_ep,f'videos/{camera}/from_timestamp':float(episode[f'videos/{camera}/from_timestamp']),f'videos/{camera}/to_timestamp':float(episode[f'videos/{camera}/to_timestamp'])})
            rows.append(row)
            provenance.append(dict(episode_index=new_ep,source=str(source),source_episode=old_ep,frames=n))
            states, targets = values['observation.state'], values['action']
            indices = np.minimum(np.arange(n)[:,None]+np.arange(30)[None,:],n-1)
            rs = Rotation.from_rotvec(states[:,3:6].astype(np.float64)).as_matrix()
            rt = Rotation.from_rotvec(targets[:,3:6].astype(np.float64)).as_matrix()
            rel = np.empty((n,30,7),dtype=np.float32)
            rel[:,:,:3] = np.einsum('nji,nhj->nhi',rs,targets[indices,:3]-states[:,None,:3])
            rotations = np.einsum('nji,nhjk->nhik',rs,rt[indices])
            rel[:,:,3:6] = Rotation.from_matrix(rotations.reshape(-1,3,3)).as_rotvec().reshape(n,30,3)
            rel[:,:,6] = targets[indices,6]
            for t in sorted(set([0,n//2,n-1])):
                np.testing.assert_allclose(rel[t],chunk_to_relative(states[t],targets[indices[t]]),atol=3e-6,rtol=1e-5)
                restored = relative_chunk_to_absolute(states[t],rel[t])
                np.testing.assert_allclose(restored[:,:3],targets[indices[t],:3],atol=2e-6)
                np.testing.assert_allclose(Rotation.from_rotvec(restored[:,3:6]).as_matrix(),Rotation.from_rotvec(targets[indices[t],3:6]).as_matrix(),atol=3e-6)
            relative_rows.append(rel.reshape(-1,7));offset += n
            if new_ep % 25 == 0: print(f'prepared {new_ep+1} episodes / {offset} frames',flush=True)
    write_episodes(Dataset.from_list(rows),output)
    write_stats({k:statistics(np.concatenate(v)) for k,v in all_values.items()},output)
    write_info(dict(codebase_version='v3.0',robot_type='cobot_magic',total_episodes=len(rows),total_frames=offset,total_tasks=1,chunks_size=1000,fps=30,splits={'train':f'0:{len(rows)}'},data_path=DEFAULT_DATA_PATH,video_path=DEFAULT_VIDEO_PATH,features=features,data_files_size_in_mb=100,video_files_size_in_mb=500,action_space='eef_absolute_targets_for_anchor_relative_runtime',eef_conversion={'frame':'gripper_center','orientation':'axis_angle_rotation_vector_rad'}),output)
    rel = np.concatenate(relative_rows)
    stats = dict(contract=dict(type='groot_chunk_relative_eef',reference='observation.state[t]',formula='inverse(T_state_t) @ T_action_t_plus_h',chunk_size=30,frame='gripper_center',rotation='axis_angle_rotation_vector_rad',gripper='absolute_target_command'),samples=offset,action_rows=len(rel),action=statistics(rel))
    (output/'meta/relative_eef_action_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    (output/'meta/source_manifest.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print(f'COMPLETE episodes={len(rows)} frames={offset} relative_rows={len(rel)}; sampled native-transform and round-trip checks PASS',flush=True)


if __name__ == '__main__': main()
