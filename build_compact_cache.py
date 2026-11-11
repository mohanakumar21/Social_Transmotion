import gc
import os
import pickle
import time

import numpy as np

from dataset_jta_compact import COMPACT_DIR, KEEP_SLOTS, NUM_FRAMES, NUM_SLOTS, compact_path

FULL_DIR = os.path.join("data", "jta_all_visual_cues", "cache")
os.makedirs(COMPACT_DIR, exist_ok=True)

for split in ["val", "test", "train"]:
    t0 = time.time()
    with open(os.path.join(FULL_DIR, f"jta_{split}.pkl"), "rb") as f:
        data = pickle.load(f)
    print(f"{split}: loaded full cache ({len(data)} scenes) in {time.time() - t0:.0f}s", flush=True)

    compact = []
    for i in range(len(data)):
        joints, mask = data[i]
        data[i] = None  # release the big float64 arrays as we go
        assert joints.shape[1] == NUM_FRAMES and joints.shape[2] == NUM_SLOTS and joints.shape[3] == 4, joints.shape
        assert bool((mask == 1).all()), "mask is not all ones"
        compact.append(np.ascontiguousarray(joints[:, :, KEEP_SLOTS, :3], dtype=np.float32))
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1} scenes converted", flush=True)
    del data, joints, mask
    gc.collect()

    out = compact_path(split)
    with open(out + ".tmp", "wb") as f:
        pickle.dump(compact, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(out + ".tmp", out)
    print(f"{split}: saved {out} ({os.path.getsize(out) / 1e9:.2f} GB), {time.time() - t0:.0f}s total", flush=True)
    del compact
    gc.collect()