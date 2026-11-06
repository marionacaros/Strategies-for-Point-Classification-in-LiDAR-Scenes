"""
Uncertainty-based inference (the paper's main strategy).

Every window is predicted once. If its predictions are not unanimous, the points are predicted again with a
different random grouping; points whose votes disagree among the non-ground classes are "uncertain" and are
predicted once more on N_SAMPLES clouds drawn around them with utils.knn_exp_prob. Votes are accumulated per
tile, indexed by point ID, so that overlapping windows are merged. IoUs are reported per --in_path directory.
"""
import argparse
import glob
import json
import logging
import os
import time

import numpy as np
import torch
from progressbar import progressbar
from sklearn.metrics import confusion_matrix

from src.config import N_CLASSES
from src.datasets import CAT3SamplingDataset
from utils.get_metrics import append_results_csv, get_ious, print_ious
from utils.utils import get_sampled_sequence, knn_exp_prob, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOTES = 3
N_SAMPLES = 64  # KNN samples
INITIAL_TILE_BUFFER = 1_000_000  # rows of the per-tile vote buffer; grows with the largest point ID


def test(number_of_workers, model, list_files, n_points):
    """
    Predict all the windows of one tile.

    :return: (targets, preds) numpy arrays with one entry per point of the tile that received a prediction
    """
    count_uncertain = 0
    count_sure_pc = 0
    count_no_voting = 0

    test_dataset = CAT3SamplingDataset(number_of_points=n_points, files=list_files, tile_ids=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                  num_workers=number_of_workers, drop_last=False)

    # Per-tile buffers indexed by point ID; they grow to the largest point ID seen
    preds_tile = torch.zeros((INITIAL_TILE_BUFFER, N_CLASSES), dtype=torch.int64)  # one-hot vote counts
    labels_tile = np.full(INITIAL_TILE_BUFFER, -1, dtype=np.int64)  # -1 = point not seen

    with torch.no_grad():
        for data in progressbar(test_dataloader):
            pointcloud, targets, _, ids, N = data  # [1, n_groups, n_points, dims]
            ids_ini = ids.squeeze(0).cpu().to(torch.int64)

            needed = int(ids_ini.max()) + 1
            if needed > preds_tile.shape[0]:
                new_size = max(needed, 2 * preds_tile.shape[0])
                preds_tile = torch.cat((preds_tile,
                                        torch.zeros((new_size - preds_tile.shape[0], N_CLASSES), dtype=torch.int64)))
                labels_tile = np.concatenate((labels_tile,
                                              np.full(new_size - labels_tile.shape[0], -1, dtype=np.int64)))

            targets = targets.view(-1).cpu().numpy()

            pc = pointcloud.squeeze(0).to(device)  # [n_groups, n_points, dims]
            dims = pc.shape[2]
            dupli = False
            knn_smpl = False  # set once the uncertain points are known; the next vote resamples around them
            uncertain_ids = set()

            if n_points < pc.shape[0] * pc.shape[1] < 32 * n_points:
                # duplicate points with a second random grouping
                pc2, ids2 = get_sampled_sequence(pc.view(-1, dims), ids_ini, n_points)
                pc = torch.cat((pc, pc2), dim=0)
                ids = torch.cat((ids_ini, ids2), dim=0)
                dupli = True
            else:
                ids = ids_ini

            for v in range(VOTES):
                if knn_smpl:
                    if len(uncertain_ids) <= 1:
                        count_sure_pc += 1
                        break
                    count_uncertain += 1
                    mask = torch.isin(ids, torch.tensor(list(uncertain_ids), dtype=torch.int64))

                    # KNN sampling only if less than 25% of the points are uncertain
                    if 1 < mask.sum() < n_points / 4:
                        knn_indices = knn_exp_prob(pc[:, :2], mask, n_points=n_points, num_samples=N_SAMPLES)
                        pc = torch.cat([pc.unsqueeze(0)[:, index, :] for index in knn_indices], dim=0)
                        ids = torch.cat([ids[index] for index in knn_indices], dim=0)
                    else:
                        pc, ids = get_sampled_sequence(pc, ids, n_points)  # [n_groups, n_points, dims]

                elif v == 1:
                    pc, ids = get_sampled_sequence(pc, ids, n_points)  # [n_groups, n_points, dims]

                preds, _ = model(pc.transpose(2, 1))  # [n_groups, 7, n_points] -> [n_groups, n_points, n_classes]
                preds = preds.contiguous().view(-1, N_CLASSES).argmax(1)
                preds = torch.nn.functional.one_hot(preds.cpu(), num_classes=N_CLASSES).to(torch.int64)

                pc = pc.reshape(-1, dims)  # [points, dims]

                preds_tile.scatter_add_(0, ids.unsqueeze(1).expand(-1, N_CLASSES), preds)
                ids_np = ids.numpy()

                if knn_smpl:
                    break

                if v == 0:
                    labels_tile[ids_ini.numpy()] = targets

                    # stop if the window is a single group or all its points got the same class
                    all_rows_same_c = (torch.sum(preds_tile[ids_np], dim=0) > 0).sum() == 1
                    if pc.shape[0] == n_points or all_rows_same_c:
                        count_no_voting += 1
                        break

                    # keep the unique points
                    pc = pc[:N, :]
                    ids = ids[:N]

                if v == 1 or dupli:
                    # uncertain = votes for more than one non-ground class (ground disagreements are ignored)
                    classes_per_point = np.count_nonzero(preds_tile[ids_np][:, 1:], axis=1)
                    uncertain_ids = set(ids_np[classes_per_point > 1])
                    knn_smpl = True

                    if v == 1:
                        pc = pc[:N, :]
                        ids = ids[:N]

            del preds, pc
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # score only the points that received at least one prediction
    predicted = (preds_tile.sum(1) != 0).numpy()
    preds_arr = torch.argmax(preds_tile, dim=1).numpy()[predicted]
    targets_arr = labels_tile[predicted]

    print(f'windows without voting: {count_no_voting}')
    print(f'windows without uncertain points: {count_sure_pc}')
    print(f'windows with uncertain points: {count_uncertain}')

    return targets_arr, preds_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='src/results/test_metrics/knn_test_results',
                        help='output directory')
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--n_points', type=int, default=8000, help='number of points per point cloud')
    parser.add_argument('--model_checkpoint', type=str, default='src/checkpoints/seg_01-23-10:55_weighted.pth',
                        help='models checkpoint path')
    parser.add_argument('--in_path', type=str, nargs='+', required=True,
                        help='one or more directories of preprocessed .pt windows')
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f'Device: {device}, votes: {VOTES}, KNN samples: {N_SAMPLES}')

    model = load_model(args.model_checkpoint, N_CLASSES, device)
    model_name = os.path.basename(args.model_checkpoint).split(':')[0]
    os.makedirs(args.output_dir, exist_ok=True)

    for in_path in args.in_path:
        start_time = time.time()
        test_files = glob.glob(os.path.join(in_path, '*.pt'))
        dataset = f'knn_{N_SAMPLES}s_{model_name}_{os.path.basename(os.path.normpath(in_path))}'
        tiles = sorted({f.split('_')[-2] for f in test_files})
        logging.info(f'{dataset}: {len(tiles)} tiles')

        targets_all, preds_all = [], []
        for tile in tiles:
            print(f'--------- TILE: {tile} ---------')
            # exact match on the tile token (a substring glob could mix tiles whose names contain each other)
            tile_files = [f for f in test_files if f.split('_')[-2] == tile]
            targets, preds = test(args.n_workers, model, tile_files, args.n_points)
            targets_all.append(targets)
            preds_all.append(preds)
        targets_arr = np.concatenate(targets_all)
        preds_arr = np.concatenate(preds_all)

        iou = get_ious(targets_arr, preds_arr)
        time_min = round((time.time() - start_time) / 60, 3)
        print('--- GLOBAL METRICS ------------------')
        print_ious(iou)
        append_results_csv(os.path.join(args.output_dir, f'IoU-results-{model_name}.csv'),
                           dataset, args.n_points, iou, time_min)

        cm = confusion_matrix(targets_arr, preds_arr, labels=list(range(N_CLASSES)))
        print(f'confusion matrix (rows: target, cols: prediction, classes 0-{N_CLASSES - 1}):\n{cm}\n')
        with open(os.path.join(args.output_dir, f'confusion_matrix_{dataset}.txt'), 'w') as filehandle:
            json.dump(cm.tolist(), filehandle)

        print(f'--- TOTAL TIME: {time_min} min ---')
