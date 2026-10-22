"""Extract contiguous human-command segments without modifying raw recordings."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(args.raw_root.glob('*.raw.h5'))
    if len(sources) != 5:
        raise ValueError(f'Expected exactly five corrections, found {len(sources)}')
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = []
    for source in sources:
        with h5py.File(source, 'r') as raw:
            if not raw.attrs.get('completed') or raw.attrs['outcome'] != 'success':
                raise ValueError(f'Unfinished or unsuccessful recording: {source}')
            mask = ((raw['frames/control_source'][:] == 2)
                    & (raw['frames/is_human_intervention'][:] == 1)
                    & (raw['frames/executed_action_valid'][:] == 1))
            indices = np.flatnonzero(mask)
            if len(indices) < 30 or np.any(np.diff(indices) != 1):
                raise ValueError(f'Human segment is too short or discontinuous: {source}')
            start, end = int(indices[0]), int(indices[-1]) + 1
            commands = raw['frames/executed_action'][start:end]
            if not np.isfinite(commands).all():
                raise ValueError(f'Invalid command values: {source}')
            if not np.array_equal(commands, raw['frames/puppet_command'][start:end]):
                raise ValueError(f'Executed commands differ from puppet commands: {source}')
            target = args.output_root / source.name
            with h5py.File(target, 'w') as out:
                for key, value in raw.attrs.items():
                    out.attrs[key] = value
                out.attrs['num_frames'] = len(indices)
                out.attrs['source_raw_path'] = str(source.resolve())
                out.attrs['source_start_index'] = start
                out.attrs['source_end_index_exclusive'] = end
                out.attrs['extraction_rule'] = 'contiguous human intervention with valid executed command; tail retained'
                for group_name in ('frames', 'images'):
                    def copy_dataset(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            parent, _, leaf = name.rpartition('/')
                            group = out.require_group(group_name + ('/' + parent if parent else ''))
                            dataset = group.create_dataset(leaf, data=obj[start:end], dtype=obj.dtype)
                            for key, value in obj.attrs.items():
                                dataset.attrs[key] = value
                    raw[group_name].visititems(copy_dataset)
            record = dict(source=str(source.resolve()), output=str(target.resolve()),
                          start=start, end=end, frames=len(indices), task=str(raw.attrs['task']))
            manifest.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    (args.output_root / 'extraction_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
