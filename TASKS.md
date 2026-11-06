# TASKS

Status after the fix pass on 2026-11-06 (conda env `pytorch10`: Python 3.10, torch 2.5.1). Run `pytest -q` (8 smoke tests, ~15 s on CPU) to check that the pipeline still works end to end.

## Open — need your input or data

- [ ] **Class weights of the "weighted" model.** `seg_01-23-10:55_weighted.pth` stores the weights `[0.1718, 0.2430, 0.2430, 0.0992, 0.2430]`, but the per-class point counts and method that produced them are lost. The paper doesn't give them either. For now, training takes them directly with `--class_weights 0.1718 0.2430 0.2430 0.0992 0.2430`. If you want `WEIGHING_METHOD` to work, fill in `SAMPLES_X_CLASS_CAT3` in `src/config.py`.
- [ ] **Reproduce the paper numbers on the full test sets (needs a GPU).** Run `src/knn_test_segmentation.py` on the Alt Empordà, Ribera del Ter and Terra Alta test windows and compare with the README / paper (mIoU 94.24%). The reconstructed `src/datasets.py` has only been checked on a B23 subset, see "Verification" below. The old test-set paths (`/mnt/QPcotLIDev01/LiDAR/DL_preproc/100x100_s80_p8k_id/test/{EMP,RIB,B29}`) are not reachable from this machine.
- [ ] **`utils/utils_plot.py`** was copied from another project (6-class CAT3 colours, `plot_predictions`). Nothing calls it, since all plotting calls in the scripts are commented out. Keep it, adapt it to the 5 classes, or delete it together with the `from utils.utils_plot import *` lines.
- [ ] **NDVI range.** The paper says "features are normalized in [0, 1]", but NDVI is fed in its raw [-1, 1] range, which is what the checkpoint expects. Only the paper text could be updated here; the code should stay as it is.

## Decisions taken (documented in code and CLAUDE.md)

- `get_iou_np` keeps mapping IoU = 0 to NaN, to reproduce the paper's numbers (docstring added).
- The KNN uncertainty check keeps ignoring vote disagreements that involve class 0 (ground), as in the paper.
- mIoU excludes ground everywhere, matching the README (94.24 = mean of tower, lines, wind turbine, surrounding). `test_segmentation_pn2.py` used to include ground; it no longer does.
- `src/datasets.py` and `src/config.py` were replaced with reconstructed CAT3 versions. The files you added were from the Barlow Twins / DALES project and could not be imported.
- HAG units checked on a real B29 LAS: `Distance` is in mm (median 290), so `Distance / (1000 * max_z)` is correct.

## Done

Missing code / setup
- [x] `src/datasets.py`: `CAT3Dataset` and `CAT3SamplingDataset` rewritten. The input layout (x, y, HAG, I, G, B, NDVI) was chosen by testing candidate layouts against the best checkpoint on real B23 windows: tower IoU 0.92 / lines 0.95 with this layout, vs ≤ 0.72 / near 0 for the alternatives. Point IDs are returned as exact int64.
- [x] `src/config.py` trimmed to what this repo uses (`N_CLASSES`, `COLOR_DROPOUT`, `WEIGHING_METHOD`, `SAMPLES_X_CLASS_CAT3`).
- [x] Train script used the undefined `CostaDataset` → `CAT3Dataset`.
- [x] `test_segmentation_pn2.py` imported the non-existent `src.prod.train_segmentation_pn2` → removed, since the helpers come from `utils.utils`.
- [x] `requirements.txt`, `pytest.ini`, `.gitignore` and `tests/test_smoke.py` added. `codecarbon` is no longer required, because its only use was commented out.

Bugs
- [x] All scripts: `torch.load(checkpoint)` without `map_location` crashed on CPU-only machines (the checkpoints were saved on CUDA).
- [x] KNN / voting: `--in_path` was iterated character by character → `nargs='+'`. The hardcoded `/mnt/...` defaults were removed.
- [x] KNN: crashed on checkpoints whose `weights` is `None`.
- [x] KNN: point IDs were cast to float32 in `torch.isin`, which is wrong above 2^24. Real B23 IDs go up to 19.5M, so this affected real data → int64.
- [x] KNN: all 60M buffer rows were scored, so empty rows counted as correct ground → only points that received a prediction are scored.
- [x] KNN: 2 × 60M × 5 int64 buffers (~4.8 GB) preallocated per tile → the buffers start at 1M rows and grow to the largest point ID.
- [x] KNN: `knn_exp_prob(..., n_points=8000)` was hardcoded → uses `--n_points`.
- [x] KNN: tiles were selected with a substring glob → exact match on the tile token.
- [x] Input channels: the dataset now returns exactly the 7 model channels, so the scripts no longer strip a trailing column (the KNN script never did).
- [x] `utils.get_sampled_sequence` crashed when a cloud had fewer than n_points/2 points (the padding was too short).
- [x] Preprocessing: `--stride` was a string (TypeError in `range()`).
- [x] Preprocessing: `DATASET_NAME='EMP'` and the `/dades/.../LAS_CAT3_HAG/` path were hardcoded → `--dataset_name` (default `CAT3`) and `--hag_path` (optional; without it HAG comes from `Distance`).
- [x] Preprocessing: an invalid `--in_path` crashed with `NameError` → clean exit with a message.
- [x] Preprocessing: settings reached the workers only through `fork` → Pool initializer, also tested with `spawn`.
- [x] Train: the default `--model_checkpoint` pointed to a missing file and triggered resuming → defaults to `None`.
- [x] Train: `random.sample(range(0, 8000), 4096)` was hardcoded → `--n_train_points` (default 4096), drawn from the actual window size.

Cleanup
- [x] Checkpoint key `color_dropour` → `color_dropout` for new checkpoints. Nothing reads it, and old checkpoints keep the old key.
- [x] Run-name suffix `_buildCOSTA20w` → `--run_name`.
- [x] `CosineAnnealingLR(T_max=80)` → `T_max = epochs`.
- [x] Power-line IoU is now logged to TensorBoard.
- [x] Test list names → `--test_lists`. Results CSVs now have a header row and the same column order in both scripts.
- [x] Voting script: opened the CSV once per point → one buffered write per window, one row per point (padded duplicates no longer produce repeated rows).

## Verification
- `pytest -q`: 8 passed (synthetic LAS → preprocessing → datasets → checkpoint forward → train step → KNN inference).
- CLI runs on real data (B23 tile `ETehpt421601`, 30 windows, CPU, best checkpoint):

  | Script | Tower IoU | Surrounding IoU | mIoU (no ground) | Time |
  |---|---|---|---|---|
  | `knn_test_segmentation.py` | 90.7 | 95.7 | 93.2 | 5.4 min |
  | `test_segmentation_pn2.py` (single pass) | 87.7 | 95.2 | 91.5 | 0.9 min |

  - `test_segmentation_voting_rdm_sampling_pn2.py` (13 votes, 3 windows) wrote one CSV row per unique point (101,336 rows = 101,336 points).
  - `train_segmentation_pn2.py` ran 1 epoch on 24/6 windows and saved its checkpoint and TensorBoard logs. These were deleted afterwards.
  - `proc_no_ground.py` CLI: works with `fork` and `spawn`.
- [ ] **Look at ground IoU on B23 (35–48%).** Many B23 points are LAS class 1 (unclassified), which is labelled "surrounding", and the model predicts ~31K of them as ground. This could be a labelling difference in B23 rather than a code problem. It doesn't affect the reported mIoU, which excludes ground. Worth checking on the paper's test sets.
- Lines show as `nan` on this subset: there are only 30 line points, all predicted as tower, so the IoU is 0 and becomes NaN (the intended `get_iou_np` behaviour).
