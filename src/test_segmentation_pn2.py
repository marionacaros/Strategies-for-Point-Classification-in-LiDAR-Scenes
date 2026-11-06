"""Single-pass inference: every point of each window is predicted once."""
import argparse
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
from utils.utils import load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(output_dir, number_of_workers, model, list_files, dataset, n_points, results_csv):
    start_time = time.time()

    test_dataset = CAT3SamplingDataset(number_of_points=n_points, files=list_files)
    logging.info(f'Total samples: {len(test_dataset)}')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                  num_workers=number_of_workers, drop_last=False)

    preds_all, targets_all = [], []
    with torch.no_grad():
        for pointcloud, targets, _, ids, _ in progressbar(test_dataloader):
            pc_t = pointcloud.squeeze(0).transpose(2, 1)  # [n_groups, dims, n_points]
            logits, _ = model(pc_t.to(device))  # [n_groups, n_points, n_classes]
            preds = logits.contiguous().view(-1, N_CLASSES).argmax(1).cpu()

            # keep one prediction per point (the last group is padded with repeated points)
            _, index = np.unique(ids.view(-1).numpy(), return_index=True)
            preds_all.append(preds[index].numpy())
            targets_all.append(targets.view(-1)[index].numpy())

    predictions_array = np.concatenate(preds_all)
    targets_array = np.concatenate(targets_all)

    iou = get_ious(targets_array, predictions_array)
    time_min = round((time.time() - start_time) / 60, 3)
    print_ious(iou)
    append_results_csv(results_csv, dataset, n_points, iou, time_min)

    cm = confusion_matrix(targets_array, predictions_array, labels=list(range(N_CLASSES)))
    print(f'confusion matrix (rows: target, cols: prediction, classes 0-{N_CLASSES - 1}):\n{cm}\n')
    with open(os.path.join(output_dir, 'confusion_matrix_' + dataset + '.txt'), 'w') as filehandle:
        json.dump(cm.tolist(), filehandle)

    print(f'--- TOTAL TIME: {time_min} min ---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='src/results/test_metrics',
                        help='output directory')
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--n_points', type=int, default=4000, help='number of points to sample from original point cloud')
    parser.add_argument('--model_checkpoint', type=str, default='src/checkpoints/seg_01-23-10:55_weighted.pth',
                        help='models checkpoint path')
    parser.add_argument('--path_list_files', type=str, required=True,
                        help='directory containing the test list files')
    parser.add_argument('--test_lists', type=str, nargs='+',
                        default=['test_seg_cat3_files.txt', 'test_seg_rib_files.txt', 'test_seg_b29_files.txt'],
                        help='text files (in path_list_files) with one .pt path per line; '
                             'the third "_"-separated token of the name is used as dataset name')
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f'Device: {device}')

    model = load_model(args.model_checkpoint, N_CLASSES, device)
    model_name = os.path.basename(args.model_checkpoint).split(':')[0]
    os.makedirs(args.output_dir, exist_ok=True)

    for test_f in args.test_lists:
        with open(os.path.join(args.path_list_files, test_f), 'r') as f:
            test_files = f.read().splitlines()

        dataset = test_f.split('_')[2] + '_' + model_name
        logging.info(f'Dataset: {dataset}')
        test(args.output_dir, args.n_workers, model, test_files, dataset, args.n_points,
             os.path.join(args.output_dir, f'IoU-results-{model_name}.csv'))
