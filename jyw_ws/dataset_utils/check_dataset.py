data_home="data/rclone-v1.68.2-linux-amd64/data/9f22b6b23ed64ced8fc8074d454e5672"
h5_path=f"{data_home}/s1a0a654702c4a5482b6895b3c1ad480/s1a0a654702c4a5482b6895b3c1ad480.h5"

from lerobot.datasets.lerobot_dataset import LeRobotDataset

import json
import h5py
import numpy as np

def walk_h5(f: h5py.File):
    """打印 H5 的层级结构 + dataset 形状/类型"""
    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[DSET] {name:60s} shape={obj.shape} dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"[GRP ] {name}")
    f.visititems(_visit)

def read_dataset(f: h5py.File, key: str, slc=None):
    """读取某个 dataset：key 是路径，如 'observations/rgb'"""
    d = f[key]
    if slc is None:
        arr = d[()]              # 读全部
    else:
        arr = d[slc]             # 按切片读（比如 [:10]）
    # h5py 返回可能是 numpy 标量/数组
    return np.array(arr)

with h5py.File(h5_path, "r") as f:
    print("=== H5 STRUCTURE ===")
    walk_h5(f)

    x = f["metadata.json"][()]

    print("raw type:", type(x))

    if isinstance(x, bytes):
        s = x.decode("utf-8", errors="ignore")
    elif isinstance(x, np.void):
        s = bytes(x.tobytes()).decode("utf-8", errors="ignore")
    elif isinstance(x, str):
        s = x
    else:
        s = str(x)

    print("----- raw text -----")
    print(s[:1000])   # 先看前 1000 个字符
    print("----- end raw -----")

    try:
        meta = json.loads(s)
        print("\nJSON keys:", meta.keys())
    except Exception as e:
        print("\nNot valid JSON:", e)

    # 示例：读一个你关心的 dataset（把 key 换成你文件里实际存在的）
    # key = "actions"
    # x = read_dataset(f, key, slc=slice(0, 5))
    # print(key, x.shape, x.dtype)
    # print(x)

    d = f["cameras/head/color/data"]

    print("dataset dtype:", d.dtype)
    print("dataset shape:", d.shape)

    x = d[0]   # 只取第 1 帧
    print("\n--- element info ---")
    print("type:", type(x))
    print("dtype:", getattr(x, "dtype", None))
    print("shape:", getattr(x, "shape", None))

    if isinstance(x, np.ndarray):
        print("ndim:", x.ndim)
        print("len:", len(x))
        if x.ndim == 1:
            print("first 10 values:", x[:10])
        elif x.ndim == 3:
            print("pixel sample:", x[0, 0, :])

    
    import h5py
    import io
    import numpy as np
    from PIL import Image

    arr = f["cameras/hand_left/color/data"][0]  # 第 1 帧
    # img = Image.open(io.BytesIO(arr.tobytes())).convert("RGB")
    # print("width, height =", img.size)     # (W, H)
    # img.save("./head_000_decoded.jpg")
    import cv2

    hwc = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR, HWC
    cv2.imwrite("image.jpg", hwc)

