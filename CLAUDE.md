# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for the paper "Effective Strategies for Point Classification in LiDAR Scenes" (Remote Sensing, 2024): semantic segmentation of airborne LiDAR (ALS, RGB+NIR) point clouds with PointNet++, focusing on training/inference strategies rather than architecture. Best checkpoint: `src/checkpoints/seg_01-23-10:55_weighted.pth`.

Dependencies are in `requirements.txt`. On the dev machine use the conda env `pytorch10` (`/home/m.caros/anaconda3/envs/pytorch10/bin/python`); the base env lacks torch. In that env, import `torch` before numpy/laspy in new entry points, otherwise MKL can fail with `undefined symbol: __kmpc_global_thread_num`. Open issues and decisions are tracked in `TASKS.md`.

`src/datasets.py` and `src/config.py` were reconstructed in 2026-11 (the originals were lost). The input layout was validated against the best checkpoint on real B23 windows, but training hyperparameters such as class counts for weighting are not recovered (see TASKS.md).

## Commands

All scripts use package-style imports (`from src...`, `from utils...`), so run them from the repo root with the root on `PYTHONPATH`. `pytest.ini` already sets this for tests.

```bash
# Smoke tests (synthetic LAS -> preprocess -> datasets -> checkpoint forward -> train step -> KNN inference), ~15 s on CPU
pytest -q
pytest -q tests/test_smoke.py::test_knn_inference     # single test

# Preprocess LAS tiles -> windowed .pt tensors
PYTHONPATH=. python data_proc/proc_no_ground.py --in_path <dir_of_las_or_file_list> --out_path <out_dir> \
    --dataset_name B29 --n_points 8000 --w_size 100 --stride 50 [--hag_path <dir_with_HeightAboveGround_las>]

# Train (80/20 random train/val split of *.pt in path_list_files)
PYTHONPATH=. python src/train_segmentation_pn2.py --path_list_files <pt_dir> --run_name myrun \
    [--class_weights 0.1718 0.2430 0.2430 0.0992 0.2430] [--model_checkpoint <resume.pth>]

# Uncertainty/KNN inference (paper's main strategy), per-tile IoU; --in_path takes several dirs
PYTHONPATH=. python src/knn_test_segmentation.py --in_path <pt_dir> [<pt_dir> ...]
# Other inference strategies
PYTHONPATH=. python src/test_segmentation_pn2.py --path_list_files <dir_with_test_lists> [--test_lists a.txt b.txt]
PYTHONPATH=. python src/test_segmentation_voting_rdm_sampling_pn2.py --in_path <pt_dir> [...]
```

Training writes checkpoints to `src/checkpoints/seg_<date>_<run_name>.pth` and TensorBoard logs to `src/runs/prod/`. Results CSVs (with a header row, same column order in all scripts) go to `--output_dir`.

## Architecture / data flow

1. **Preprocessing** (`data_proc/proc_no_ground.py`): reads LAS with laspy, drops noise and overlap classes, maps LAS 8→2 (ground), and splits tiles into `w_size`×`w_size` m windows with `stride`. It removes ground when there are enough non-ground points, otherwise it keeps just enough ground to reach `n_points`. It also computes NDVI. Each window is saved as a `.pt` tensor `[N, 12]` with columns `x,y,z,class,I,R,G,B,NIR,NDVI,HAG,point_id`. HAG = LAS `Distance` (mm) / (1000·max_z), or `HeightAboveGround / max_z` from `--hag_path`. `point_id` is unique within a LAS tile (it can exceed 2^24, so keep IDs as int64, never float32). File names are `<prefix>_<dataset>_<lasname>_w<idx>.pt`; inference groups windows by tile via `path.split('_')[-2]`. Worker settings are passed with a Pool initializer.
2. **Datasets** (`src/datasets.py`): model input is **7 channels: `x, y` (normalized to [-1,1] per window), `HAG, I, G, B, NDVI`**. Labels: `0` ground (LAS 2), `1` tower (15), `2` lines (14), `3` surrounding (everything else), `4` wind turbine (29, 19). `CAT3Dataset` returns `(pc [n,7], labels [n], filename)` with a fixed number of points. `CAT3SamplingDataset` keeps every point and returns `(pc [groups, n_points, 7], labels, filename, ids int64, n_unique)`; the first `n_unique` flattened rows are the unique points. `tile_ids=True` uses column 11 as IDs, which the KNN script needs to merge overlapping windows.
3. **Model** (`src/models/pointnet2_sem_seg.py` + `pointnet2_utils.py`): PointNet++ SSG with 4 set-abstraction + 4 feature-propagation layers. Input is `[B, 7, N]` with N ≥ 1024. `sample_and_group` prepends only the normalized z offset, which is why `sa1` has `in_channel=8`. Output is **log-softmax** `[B, N, C]` plus `None`. `src/models/pointnet.py` is the older PointNet baseline.
4. **Training** (`src/train_segmentation_pn2.py`): draws `--n_train_points` (4096) random points from each window per step, applies z-rotation and color dropout (zeroes channels `4:` = G, B, NDVI with prob `COLOR_DROPOUT`), and uses AdamW + cosine LR (`T_max = epochs`) with CE loss (class weights from `--class_weights` or `WEIGHING_METHOD`). The checkpoint is saved on best val loss. The checkpoint dict (`utils/utils.save_checkpoint`) holds `model`, `optimizer`, `weights`, `batch_size`, `lr`, `number_of_points`, `epoch`, etc., and the inference scripts read these keys.
5. **Inference strategies**:
   - `src/test_segmentation_pn2.py`: single pass.
   - `src/test_segmentation_voting_rdm_sampling_pn2.py`: `VOTES` random resamplings, writes per-point vote CSVs.
   - `src/knn_test_segmentation.py`: uncertainty-based. It accumulates one-hot votes in a per-tile buffer indexed by point ID (the buffer grows as needed). Points whose votes disagree among classes 1–4 are "uncertain"; ground disagreements are ignored on purpose, as in the paper. Uncertain points are resampled with `utils.knn_exp_prob` (exponential-distance sampling in xy).

## Metrics

The reported mIoU **excludes ground**: it is the mean of tower, lines, wind turbine and surrounding, which matches the README's 94.24%. `utils/get_metrics.get_iou_np` intentionally returns NaN for an IoU of 0, so a fully missed class is skipped by `nanmean`. This is kept to reproduce the paper's numbers; don't "fix" it without asking.
