import os as _os
from pathlib import Path as _Path
REPO_ROOT = _Path(__file__).resolve().parents[2]
WS_ROOT = REPO_ROOT.parent
OUTPUTS = REPO_ROOT / "outputs"
DATASETS = WS_ROOT / "datasets"
import os
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
import numpy as np, torch, json
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.relative_eef import relative_chunk_to_absolute
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
import sys; sys.path.insert(0,'scripts')
from train_z_stage1 import ZEncoder

P=str(OUTPUTS / "cobot_magic_smolvla_metal_v4_phase1_frozen60k_bs32_gpu_256_photoaug_gw/checkpoints/060000/pretrained_model")
import os
Z=os.environ.get("Z_PATH",str(OUTPUTS / "z_stage1_v1/z_encoder.pt"))
dev=torch.device("cuda")
pol=SmolVLAPolicy.from_pretrained(P, local_files_only=True).to(dev).eval()
pre,post=make_pre_post_processors(policy_cfg=pol.config, pretrained_path=P)
vla=pol.model
ck=torch.load(Z, map_location="cpu")
cfg=json.load(open(os.environ.get("Z_CONFIG",str(OUTPUTS / "z_stage1_v1/config.json"))))
enc=ZEncoder(cfg["vlm_dim"], z_dim=cfg["z_dim"], d_model=cfg["d_model"]).to(dev).eval()
enc.load_state_dict(ck["encoder"])

ds=LeRobotDataset("local/cobot_magic_metal_v4_iv5", root=str(DATASETS / "cobot_magic_metal_v4_iv5"),
                  delta_timestamps={"action":[i/30 for i in range(30)]}, video_backend="pyav")
rng=np.random.default_rng(0)
frames=sorted(rng.choice(np.arange(0,len(ds)-35,3), 220, replace=False))
Zs=[];Y0=[];S=[];R0=[]
with torch.no_grad():
    for i in frames:
        s=ds[int(i)]; st=s["observation.state"].float()
        b=pre({k:(v.unsqueeze(0) if torch.is_tensor(v) else v) for k,v in s.items()})
        imgs,masks=pol.prepare_images(b); state=pol.prepare_state(b)
        prefix,pmask=vla.extract_rlt_prefix(imgs,masks,b[OBS_LANGUAGE_TOKENS],b[OBS_LANGUAGE_ATTENTION_MASK],state)
        z=enc(prefix,pmask)[0].float().cpu().numpy()
        p=pol.predict_action_chunk(b)
        rel=torch.stack([post(p[:,k,:]) for k in range(p.shape[1])],1).squeeze(0).float().cpu().numpy()
        ref=relative_chunk_to_absolute(st.numpy(),rel)
        human=s["action"].float().numpy()
        Zs.append(z); Y0.append((human[0]-ref[0])); S.append(st.numpy()); R0.append(ref[0])
Z=np.array(Zs);Y=np.array(Y0);St=np.array(S);R=np.array(R0)
print("samples",len(Z),"z",Z.shape,"target",Y.shape)
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
for name,X in (("z",Z),("z+state",np.hstack([Z,St])),("z+state+ref0",np.hstack([Z,St,R]))):
    Xtr,Xte,Ytr,Yte=train_test_split(X,Y,test_size=0.25,random_state=0)
    m=Ridge(alpha=1.0).fit(Xtr,Ytr); pred=m.predict(Xte)
    r2=[r2_score(Yte[:,d],pred[:,d]) for d in range(7)]
    print(f"{name:14} R2 xyz={np.mean(r2[:3]):.3f} grip={r2[6]:.3f} per-dim={np.round(r2,2)}")
