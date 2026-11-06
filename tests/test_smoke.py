"""
End-to-end smoke test on synthetic data (CPU, ~1 min):
synthetic LAS -> data_proc/proc_no_ground.py -> datasets -> checkpoint forward pass -> one train step
-> KNN uncertainty inference.

Run from the repo root:  pytest -q
"""
# torch must be imported before numpy/laspy: in some conda envs loading MKL first fails with
# "undefined symbol: __kmpc_global_thread_num" (OpenMP runtime clash)
import torch  # noqa: I001

import glob
import os
import random

import laspy
import numpy as np
import pytest

import data_proc.proc_no_ground as proc
from src.datasets import CAT3Dataset, CAT3SamplingDataset, N_FEATURES, get_labels_segmen
from src.models.pointnet2_sem_seg import PointNet2
from utils.utils import get_sampled_sequence

CHECKPOINT = 'src/checkpoints/seg_01-23-10:55_weighted.pth'
N_POINTS = 2048  # smaller than the 8000 used in the paper to keep the test fast


@pytest.fixture(scope='module')
def windows(tmp_path_factory):
    """Write a synthetic 200x200 m LAS tile and split it into .pt windows."""
    tmp = tmp_path_factory.mktemp('data')
    rng = np.random.default_rng(0)
    n = 60_000
    header = laspy.LasHeader(point_format=8, version='1.4')
    header.scales = [0.01] * 3
    header.offsets = [0, 0, 0]
    header.add_extra_dim(laspy.ExtraBytesParams(name='Distance', type=np.float32))
    las = laspy.LasData(header)
    las.x = rng.uniform(0, 200, n)
    las.y = rng.uniform(0, 200, n)
    las.z = rng.uniform(0, 50, n)
    las.classification = rng.choice([2, 3, 4, 5, 14, 15, 29], n, p=[.2, .3, .2, .2, .04, .04, .02])
    las.intensity = rng.integers(0, 5000, n)
    for colour in ('red', 'green', 'blue', 'nir'):
        setattr(las, colour, rng.integers(0, 65535, n))
    las.Distance = rng.uniform(0, 50_000, n).astype(np.float32)
    las_path = tmp / 'synthetic_123456.las'
    las.write(las_path)

    out = tmp / 'windows'
    out.mkdir()
    proc._init_worker(dict(DATASET_NAME='TEST', HAG_PATH=None, STORE_DOUBLE=True, OUT_PATH=str(out),
                           N_POINTS=N_POINTS, W_SIZE=100, MAX_Z=200.0, MAX_I=5000.0, STRIDE=50))
    proc.split_pointcloud(str(las_path))
    files = sorted(glob.glob(str(out / '*.pt')))
    assert files, 'preprocessing wrote no windows'
    return files


@pytest.fixture(scope='module')
def model():
    m = PointNet2(num_classes=5)
    m.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=True)['model'])
    return m.eval()


def test_preprocessing_layout(windows):
    pc = torch.load(windows[0], weights_only=True)
    assert pc.shape[1] == 12 and pc.shape[0] >= N_POINTS
    assert pc.dtype == torch.float64
    # tile id is the second-to-last '_' token, as the KNN script expects
    assert os.path.basename(windows[0]).split('_')[-2] == '123456'


def test_labels():
    assert get_labels_segmen(np.array([2, 15, 14, 29, 19, 3, 1, 6])).tolist() == [0, 1, 2, 4, 4, 3, 3, 3]


def test_train_dataset(windows):
    pc, labels, _ = CAT3Dataset(number_of_points=N_POINTS, files=windows)[0]
    assert pc.shape == (N_POINTS, N_FEATURES) and pc.dtype == torch.float32
    assert labels.shape == (N_POINTS,) and labels.dtype == torch.int64
    assert pc[:, :2].min() >= -1 and pc[:, :2].max() <= 1


def test_sampling_dataset_alignment(windows):
    raw = torch.load(windows[0], weights_only=True).numpy()
    pc, labels, _, ids, n_unique = CAT3SamplingDataset(number_of_points=N_POINTS, files=[windows[0]],
                                                       tile_ids=True)[0]
    assert n_unique == raw.shape[0]
    assert pc.shape[1:] == (N_POINTS, N_FEATURES)
    assert pc.shape[0] * N_POINTS == labels.shape[0] == ids.shape[0]
    assert ids.dtype == torch.int64
    # first n_unique rows are the unique points, and labels / features follow their ids
    assert len(set(ids[:n_unique].tolist())) == n_unique
    by_id = {int(i): k for k, i in enumerate(raw[:, 11])}
    rows = np.array([by_id[int(i)] for i in ids])
    assert np.array_equal(labels.numpy(), get_labels_segmen(raw[rows, 3]))
    assert np.allclose(pc.view(-1, N_FEATURES)[:, 2].numpy(), raw[rows, 10], atol=1e-6)  # HAG


def test_get_sampled_sequence_small_cloud():
    pc, ids = get_sampled_sequence(torch.rand(300, 4), torch.arange(300), 1000)
    assert pc.shape == (1, 1000, 4) and ids.shape == (1000,)
    assert set(ids.tolist()) == set(range(300))


def test_forward_with_checkpoint(windows, model):
    pc, *_ = CAT3SamplingDataset(number_of_points=N_POINTS, files=[windows[0]])[0]
    with torch.no_grad():
        out, _ = model(pc[:2].transpose(2, 1))
    assert out.shape == (2, N_POINTS, 5)
    assert torch.allclose(out.exp().sum(-1), torch.ones(1), atol=1e-4)


def test_train_step(windows):
    from src import train_segmentation_pn2 as train_script
    loader = torch.utils.data.DataLoader(CAT3Dataset(number_of_points=N_POINTS, files=windows), batch_size=2)
    m = PointNet2(num_classes=5)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    metrics, targets, preds = train_script.train_loop(next(iter(loader)), opt, loss, m, train=True,
                                                      device='cpu', n_train_points=1024)
    assert torch.isfinite(metrics['loss'])
    assert targets.shape == preds.shape == (2 * 1024,)


def test_knn_inference(windows, model):
    import src.knn_test_segmentation as knn
    knn.N_SAMPLES = 2  # fewer KNN resamplings than the paper (64) to keep CPU time low
    random.seed(0)
    torch.manual_seed(0)
    targets, preds = knn.test(0, model, windows[:3], N_POINTS)
    n_points_tile = len(set().union(*(torch.load(f, weights_only=True)[:, 11].long().tolist()
                                      for f in windows[:3])))
    # one prediction per unique point of the tile, no extra empty buffer rows
    assert len(targets) == len(preds) == n_points_tile
    assert set(np.unique(preds)) <= set(range(5))


def test_ious_exclude_ground():
    from utils.get_metrics import get_ious
    targets = np.array([0, 0, 1, 1, 3, 3])
    preds = np.array([3, 3, 1, 1, 3, 3])  # ground fully missed: must not affect the reported IoUs
    iou = get_ious(targets, preds)
    assert set(iou) == {'tower', 'lines', 'surrounding', 'wind_turbine', 'mIoU'}
    assert iou['tower'] == 1.0 and iou['surrounding'] == 0.5
    assert iou['mIoU'] == 0.75
