from types import SimpleNamespace

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.sampler import ConsecutiveFrameBatchSampler


def test_sampler_coverage_boundaries_and_seed():
    torch.manual_seed(12)
    sampler = ConsecutiveFrameBatchSampler([0, 19, 24], [19, 24, 42], 16, 4)
    batches = list(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(42))
    assert len(batches) == len(sampler) == 3
    assert all(len(batch) == 16 for batch in batches[:-1])
    for block in sampler.blocks:
        assert all(i == block[0] + k for k, i in enumerate(block))
        assert not (block[0] < 19 <= block[-1] or block[0] < 24 <= block[-1])
    torch.manual_seed(12)
    assert batches == list(sampler)
    other = ConsecutiveFrameBatchSampler([0, 19], [19, 24], 16, 4, drop_n_last_frames=2)
    assert sorted(i for batch in other for i in batch) == list(range(17)) + list(range(19, 22))


class FakeDataset:
    __getitems__ = LeRobotDataset.__getitems__

    def __init__(self):
        self.meta = SimpleNamespace(video_keys=['camera'])
        self.delta_indices = None
        self.batch_video_decode = True
        self.calls = []
        self.hf_dataset = [{'episode_index': torch.tensor(i // 10), 'index': torch.tensor(i),
                            'timestamp': torch.tensor(float(i % 10))} for i in range(20)]

    def _ensure_hf_dataset_loaded(self):
        pass

    def _get_query_timestamps(self, timestamp, queries):
        return {'camera': [timestamp]}

    def _query_videos(self, queries, episode):
        self.calls.append((episode, queries))
        return {'camera': torch.stack([torch.full((3, 2, 2), t + episode * 10)
                                       for t in queries['camera']]).squeeze(0)}

    def __getitem__(self, index):
        if getattr(self, '_batch_video_frames', None) is not None:
            return self._batch_video_frames[index]
        item = self.hf_dataset[index]
        return self._query_videos(self._get_query_timestamps(item['timestamp'].item(), None),
                                  item['episode_index'].item())


def test_batched_decode_order_groups_and_cache_cleanup():
    dataset = FakeDataset()
    indices = [13, 12, 2, 3, 7, 8]
    result = dataset.__getitems__(indices)
    assert len(dataset.calls) == 3
    assert [int(item['camera'][0, 0, 0]) for item in result] == indices
    assert all(item['camera'].shape == (3, 2, 2) for item in result)
    assert dataset._batch_video_frames is None
    dataset.batch_video_decode = False
    dataset.calls.clear()
    normal = dataset.__getitems__(indices)
    assert len(dataset.calls) == len(indices)
    assert all(torch.equal(a['camera'], b['camera']) for a, b in zip(result, normal, strict=True))
