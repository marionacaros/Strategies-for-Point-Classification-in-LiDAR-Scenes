"""
Datasets for the LiDAR-CAT3 windows written by data_proc/proc_no_ground.py.

Each window is a .pt tensor [points, 12] with columns:
    0-2 x, y, z | 3 LAS classification | 4 intensity | 5-7 R, G, B | 8 NIR | 9 NDVI |
    10 height above ground (HAG, already normalized by max_z) | 11 point ID (unique within a LAS tile)

Model input features (7 channels, in this order):
    x, y (normalized to [-1, 1] per window), HAG, intensity, G, B, NDVI
This layout was checked against src/checkpoints/seg_01-23-10:55_weighted.pth on real B23 windows
(tower IoU 0.92, lines IoU 0.95); other orderings give much worse results.

Segmentation labels (N_CLASSES = 5):
    0 ground (LAS 2; LAS 8 is mapped to 2 during preprocessing)
    1 transmission tower (LAS 15)
    2 power lines (LAS 14)
    3 surrounding (every other class: vegetation, buildings, unclassified, ...)
    4 wind turbine (LAS 29, 19)
"""
import random

import numpy as np
import torch
import torch.utils.data as data

from utils.utils import get_sampled_sequence

N_FEATURES = 7


def load_window(point_file):
    """Load a window as a float64 numpy array [points, 12]."""
    with open(point_file, 'rb') as f:
        pc = torch.load(f, map_location='cpu', weights_only=True)
    return pc.double().numpy()


def get_labels_segmen(las_classes):
    """Map raw LAS classes [points] to segmentation labels [points] (see module docstring)."""
    c = np.asarray(las_classes)
    labels = np.select([c == 2,
                        c == 15,
                        c == 14,
                        (c == 29) | (c == 19)],
                       [0, 1, 2, 4],
                       default=3)
    return labels.astype(np.int64)


def get_features(pc):
    """
    Build the 7 model input features from a raw window [points, 12].

    :return: float32 array [points, 7]: x, y, HAG, I, G, B, NDVI
    """
    feats = np.empty((pc.shape[0], N_FEATURES), dtype=np.float32)
    for out_col, axis in ((0, 0), (1, 1)):
        lo, hi = pc[:, axis].min(), pc[:, axis].max()
        feats[:, out_col] = 2 * (pc[:, axis] - lo) / (hi - lo) - 1 if hi > lo else 0.
    feats[:, 2] = pc[:, 10]                      # HAG
    feats[:, 3] = np.clip(pc[:, 4], 0.0, 1.0)    # intensity
    feats[:, 4] = np.clip(pc[:, 6], 0.0, 1.0)    # green
    feats[:, 5] = np.clip(pc[:, 7], 0.0, 1.0)    # blue
    feats[:, 6] = pc[:, 9]                       # NDVI
    return feats


class CAT3Dataset(data.Dataset):
    """
    Training dataset: one window per item, with exactly `number_of_points` points
    (random subsample, or random duplication when the window is smaller).

    Item: (pc float32 [number_of_points, 7], labels int64 [number_of_points], filename)
    """

    def __init__(self, number_of_points=8000, files=None):
        self.n_points = number_of_points
        self.files = list(files or [])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        pc = load_window(filename)

        n = pc.shape[0]
        if n > self.n_points:
            pc = pc[random.sample(range(n), self.n_points)]
        elif n < self.n_points:
            pc = np.concatenate([pc, pc[np.random.randint(0, n, self.n_points - n)]], axis=0)

        labels = torch.from_numpy(get_labels_segmen(pc[:, 3]))
        feats = torch.from_numpy(get_features(pc))
        return feats, labels, filename


class CAT3SamplingDataset(CAT3Dataset):
    """
    Evaluation dataset: every point of the window is kept. The window is shuffled and cut into
    groups of `number_of_points` (the last group is padded with repeated points), see
    utils.utils.get_sampled_sequence. The first `n_unique` rows of the flattened output are the
    unique points of the window.

    Item: (pc float32 [n_groups, number_of_points, 7],
           labels int64 [n_groups * number_of_points],
           filename,
           ids int64 [n_groups * number_of_points],
           n_unique int)

    :param tile_ids: use the point IDs stored in column 11 (unique within a LAS tile, needed to merge
        predictions of overlapping windows). Otherwise IDs are 0..n-1 within the window.
    """

    def __init__(self, number_of_points=8000, files=None, tile_ids=False):
        super().__init__(number_of_points, files)
        self.tile_ids = tile_ids

    def __getitem__(self, index):
        filename = self.files[index]
        pc = load_window(filename)
        n_unique = pc.shape[0]

        if self.tile_ids:
            ids = torch.from_numpy(pc[:, 11].astype(np.int64))
        else:
            ids = torch.arange(n_unique, dtype=torch.int64)

        # shuffle features and labels together so they stay aligned with the ids
        labels = get_labels_segmen(pc[:, 3])
        stacked = torch.from_numpy(np.concatenate([get_features(pc).astype(np.float64),
                                                   labels[:, None].astype(np.float64)], axis=1))

        stacked, ids = get_sampled_sequence(stacked, ids, self.n_points)  # [n_groups, n_points, 8]
        stacked = stacked.view(-1, stacked.shape[2])

        feats = stacked[:, :N_FEATURES].float().view(-1, self.n_points, N_FEATURES)
        labels = stacked[:, N_FEATURES].long()
        return feats, labels, filename, ids, n_unique
