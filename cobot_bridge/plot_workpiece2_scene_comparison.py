"""Compare the old training scene and new raw demonstrations at start and grasp."""
from pathlib import Path
import json
import av
import cv2
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def video_image(path: Path, frame_index: int) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        target = frame_index / 30
        container.seek(int(target / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time >= target - 1 / 60:
                return frame.to_ndarray(format="rgb24")
    raise ValueError(f"missing frame {frame_index}: {path}")


def main():
    project = Path(__file__).resolve().parents[2]
    original = project / "datasets/cobot_magic_5482_grippercmd_right_joint_v3"
    new = project / "datasets/cobot_magic_il_workpiece2_realaction_headtrim_v3"
    entry = json.loads((new / "meta/cobot_magic_conversion.json").read_text())["episodes"][0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    with h5py.File(entry["raw"]) as raw:
        state = np.asarray(raw["frames/puppet_position"])
        command = np.asarray(raw["frames/executed_action"])
        valid = np.asarray(raw["frames/executed_action_valid"], bool)
        contact = np.flatnonzero(valid & (np.abs(command[:, 6]) < .002) & (state[:, 6] > .02))
        hold = int(contact[len(contact) // 2])
        for row, (old_idx, new_idx, stage) in enumerate([(0, entry["start_index"], "Start"), (300, hold, "Object held")]):
            for col, camera in enumerate(["f", "r"]):
                path = original / f"videos/observation.images.camera_{camera}/chunk-000/file-000.mp4"
                axes[row, col].imshow(video_image(path, old_idx))
                axes[row, col].set_title(f"Old dataset | {stage} | {'head' if camera=='f' else 'right wrist'}", fontsize=10)
            for col, camera in enumerate(["front", "right"], 2):
                jpeg = np.asarray(raw[f"images/{camera}/jpeg"][new_idx], np.uint8)
                bgr = cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("invalid raw JPEG")
                axes[row, col].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                axes[row, col].set_title(f"New workpiece2 | {stage} | {camera}", fontsize=10)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Old 100k training data vs new workpiece2 demonstrations (episode 0)", fontsize=15)
    output = project / "lerobot/web_runs/workpiece2_adaptation/scene_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
