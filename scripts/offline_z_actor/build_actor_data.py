#!/usr/bin/env python3
"""Build residual-actor training arrays from the current-environment episodes.

For each sampled frame: z (frozen 60k prefix + z encoder), state, ref chunk
(60k policy), executed action chunk, and the policy/human label.
"""
import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]
WS_ROOT = REPO_ROOT.parent
OUTPUTS = REPO_ROOT / "outputs"
DATASETS = WS_ROOT / "datasets"
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys, json
import numpy as np, torch
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.relative_eef import relative_chunk_to_absolute
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_z_stage1 import ZEncoder

P = str(OUTPUTS / "cobot_magic_smolvla_metal_v4_phase1_frozen60k_bs32_gpu_256_photoaug_gw/checkpoints/060000/pretrained_model")
Z = os.environ.get("Z_PATH", str(OUTPUTS / "z_stage1_v1/z_encoder.pt"))
DS = os.environ.get("DS_ROOT", str(DATASETS / "workpiece3_full"))
OUT = os.environ.get("OUT_NPZ", str(OUTPUTS / "actor_data/workpiece3_full.npz"))
STRIDE = int(os.environ.get("STRIDE", "2"))
dev = torch.device("cuda")

pol = SmolVLAPolicy.from_pretrained(P, local_files_only=True).to(dev).eval()
pre, post = make_pre_post_processors(policy_cfg=pol.config, pretrained_path=P)
vla = pol.model
cfg = json.load(open(os.environ.get("Z_CONFIG", str(OUTPUTS / "z_stage1_v1/config.json"))))
enc = ZEncoder(cfg["vlm_dim"], z_dim=cfg["z_dim"], d_model=cfg["d_model"]).to(dev).eval()
enc.load_state_dict(torch.load(Z, map_location="cpu")["encoder"])

ds = LeRobotDataset("local/workpiece3_full", root=DS,
                    delta_timestamps={"action": [i / 30 for i in range(30)]}, video_backend="pyav")
print("frames", len(ds), "episodes", ds.num_episodes, flush=True)

zs, states, refs, acts, human, eps, frm = [], [], [], [], [], [], []
with torch.no_grad():
    for i in range(0, len(ds), STRIDE):
        s = ds[i]
        cs = int(np.asarray(s["observation.control_source"]).reshape(-1)[0])
        if cs not in (1, 2):
            continue
        st = s["observation.state"].float()
        b = pre({k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in s.items()})
        imgs, masks = pol.prepare_images(b); state = pol.prepare_state(b)
        prefix, pmask = vla.extract_rlt_prefix(imgs, masks, b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK], state)
        z = enc(prefix, pmask)[0].float().cpu().numpy()
        p = pol.predict_action_chunk(b)
        rel = torch.stack([post(p[:, k, :]) for k in range(p.shape[1])], 1).squeeze(0).float().cpu().numpy()
        ref = relative_chunk_to_absolute(st.numpy(), rel)
        zs.append(z); states.append(st.numpy()); refs.append(ref)
        acts.append(s["action"].float().numpy()); human.append(1.0 if cs == 2 else 0.0)
        eps.append(int(np.asarray(s["episode_index"]).reshape(-1)[0]))
        frm.append(int(np.asarray(s["frame_index"]).reshape(-1)[0]))
        if len(zs) % 200 == 0:
            print("built", len(zs), flush=True)

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, z=np.stack(zs), state=np.stack(states), ref=np.stack(refs),
                    action=np.stack(acts), human=np.array(human, np.float32),
                    episode=np.array(eps), frame=np.array(frm))
print("saved", OUT, "samples", len(zs), "human", int(np.sum(human)))
