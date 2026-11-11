import os
import pickle
import time

import torch

# re-exported so train_jta_pooled.py / evaluate_jta_pooled.py keep working
from dataset_jta import batch_process_coords, collate_batch  # noqa: F401

COMPACT_DIR = os.path.join("data", "jta_all_visual_cues", "cache_compact")
KEEP_SLOTS = [0] + list(range(3, 25))  # slot 0 = trajectory, slots 3..24 = the 22 3D-pose joints
NUM_SLOTS = 47
NUM_FRAMES = 21


def compact_path(split):
    return os.path.join(COMPACT_DIR, f"jta_{split}_compact.pkl")


class JtaCompactDataset(torch.utils.data.Dataset):
    """Same samples as JtaAllVisualCuesDataset, but only what the pooled model reads is stored
    (trajectory + 3D pose, float32). The 47-slot float64 layout is rebuilt per sample, with zeros
    in the slots the pooled model never reads (boxes, 2D pose)."""

    def __init__(self, split="train", track_size=NUM_FRAMES, track_cutoff=9, **unused):
        assert track_size == NUM_FRAMES, "the compact cache holds 21-frame scenes"
        t = time.time()
        with open(compact_path(split), "rb") as f:
            self.scenes = pickle.load(f)
        print(f"[compact] loaded '{split}' ({len(self.scenes)} scenes) in {time.time() - t:.1f}s")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        small = torch.from_numpy(self.scenes[idx])  # [N, 21, 23, 3] float32
        n = small.shape[0]
        joints = torch.zeros((n, NUM_FRAMES, NUM_SLOTS, 4), dtype=torch.float64)
        joints[:, :, 0, :3] = small[:, :, 0, :]
        joints[:, :, 3:25, :3] = small[:, :, 1:, :]
        mask = torch.ones((n, NUM_FRAMES, NUM_SLOTS), dtype=torch.float64)
        return joints, mask


def create_dataset(dataset_name, logger, **args):
    logger.info("Loading dataset " + dataset_name + " (compact)")
    if dataset_name == 'jta_all_visual_cues':
        return JtaCompactDataset(**args)
    raise ValueError(f"Dataset with name '{dataset_name}' not found.")


def get_datasets(datasets_list, config, logger):
    in_F, out_F = config['TRAIN']['input_track_size'], config['TRAIN']['output_track_size']
    return [create_dataset(name, logger, split="train", track_size=(in_F + out_F), track_cutoff=in_F)
            for name in datasets_list]