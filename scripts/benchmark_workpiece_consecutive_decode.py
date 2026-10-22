"""Compare independent and shared decode for identical block-sampled anchors."""
import argparse
import json
import time

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import ConsecutiveFrameBatchSampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--block', type=int, default=8)
    parser.add_argument('--loader-workers', type=int, default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    d = LeRobotDataset('local/cobot_magic_workpiece_all407_anchor_eef_v3',root=args.root,
                      delta_timestamps={'action':[i/30 for i in range(30)]},video_backend='pyav')
    torch.manual_seed(20261014)
    sampler = ConsecutiveFrameBatchSampler(d.meta.episodes['dataset_from_index'],d.meta.episodes['dataset_to_index'],32,args.block)
    if args.loader_workers is not None:
        # Do not decode in the parent before forking video workers.
        for mode in ['random_independent', 'consecutive_shared']:
            torch.manual_seed(20261014)
            d.batch_video_decode = mode == 'consecutive_shared'
            kwargs = dict(num_workers=args.loader_workers, pin_memory=False)
            if args.loader_workers:
                kwargs['prefetch_factor'] = 2
            if d.batch_video_decode:
                loader = torch.utils.data.DataLoader(d, batch_sampler=sampler, **kwargs)
            else:
                loader = torch.utils.data.DataLoader(d, batch_size=32, shuffle=True, **kwargs)
            iterator = iter(loader)
            for _ in range(5): next(iterator)
            times = []
            for _ in range(args.batches):
                start = time.perf_counter();batch = next(iterator);times.append(time.perf_counter()-start)
                assert batch['action'].shape == (32,30,7)
            print(json.dumps(dict(mode=mode,workers=args.loader_workers,batches=args.batches,
                                  mean_batch_s=float(np.mean(times)),median_batch_s=float(np.median(times)),
                                  batches_per_s=len(times)/sum(times))),flush=True)
            if args.loader_workers:
                iterator._shutdown_workers()
            del iterator, loader
        return
    iterator = iter(sampler)
    report = []
    for k in range(args.batches):
        indices = next(iterator)
        d.batch_video_decode = False
        start = time.perf_counter();normal = d.__getitems__(indices);normal_s = time.perf_counter()-start
        d.batch_video_decode = True
        start = time.perf_counter();shared = d.__getitems__(indices);shared_s = time.perf_counter()-start
        for a,b in zip(normal,shared,strict=True):
            assert a.keys() == b.keys()
            for key in a:
                if torch.is_tensor(a[key]):
                    torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                else: assert a[key] == b[key]
        result=dict(batch=k,normal_s=normal_s,shared_s=shared_s,speedup=normal_s/shared_s,
                    exact_all_fields_match=True,block=args.block)
        report.append(result);print(json.dumps(result),flush=True)
    print('MEDIAN',np.median([r['normal_s'] for r in report]),np.median([r['shared_s'] for r in report]),flush=True)


if __name__ == '__main__': main()
