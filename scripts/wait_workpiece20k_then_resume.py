"""Wait for the specific current run to finish successfully, then resume once.

If the source process is stopped early, times out or does not publish its final
checkpoint, do not start another training process. Ctrl-C cancels this queue.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pid', type=int, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--resume-script', type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 4 * 3600
    print(f'Waiting for PID {args.source_pid} and complete 020000 checkpoint; no concurrent training.', flush=True)
    while True:
        if time.monotonic() > deadline:
            raise RuntimeError('Wait expired; continuation not started')
        proc = Path(f'/proc/{args.source_pid}/cmdline')
        try:
            command = proc.read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except FileNotFoundError:
            break
        if not command:
            break
        if args.source_run.name not in command or 'accelerate' not in command:
            raise RuntimeError('Source PID changed identity; continuation not started')
        time.sleep(10)
    checkpoint = args.source_run / 'checkpoints/020000'
    last = args.source_run / 'checkpoints/last'
    if not last.is_symlink() or last.resolve() != checkpoint.resolve():
        raise RuntimeError('Source did not finish final checkpoint; continuation not started')
    step = json.loads((checkpoint/'training_state/training_step.json').read_text())
    if step.get('step') != 20000:
        raise RuntimeError(f'Unexpected checkpoint training step: {step}')
    for relative in ['pretrained_model/model.safetensors','pretrained_model/train_config.json',
                     'training_state/optimizer_state.safetensors']:
        if not (checkpoint/relative).is_file(): raise RuntimeError(f'Missing {relative}')
    print('Complete 20k checkpoint verified; starting batch32 continuation to total60k.', flush=True)
    subprocess.run(['bash', str(args.resume_script.resolve())], check=True)


if __name__ == '__main__': main()
