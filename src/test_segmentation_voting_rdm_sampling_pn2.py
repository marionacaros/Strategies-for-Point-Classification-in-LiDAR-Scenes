"""
Voting inference: every window is predicted VOTES times with different random groupings of its points.
Writes one CSV per window (preds-<window>.csv) with the label and the VOTES predictions of each point.
"""
import argparse
import glob
import logging
import os
import time

import torch
from progressbar import progressbar

from src.config import N_CLASSES
from src.datasets import CAT3SamplingDataset
from utils.utils import get_sampled_sequence, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VOTES = 13
N_POINTS = 8000


def test(output_dir, number_of_workers, model, list_files):
    """
    Iterate VOTES times over each window and store the per-point predictions in csv files.

    :param output_dir: directory where csv files are stored
    :param number_of_workers: num cpus
    :param model: PointNet2 in eval mode
    :param list_files: .pt windows
    """
    start_time = time.time()

    test_dataset = CAT3SamplingDataset(number_of_points=N_POINTS, files=list_files)
    logging.info(f'Total samples: {len(test_dataset)}')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                                  num_workers=number_of_workers, drop_last=False)

    with torch.no_grad():
        for pointcloud, targets, file_name, ids, N in progressbar(test_dataloader):
            ids = ids.squeeze(0).cpu().to(torch.int64)
            N = int(N)
            file_name = os.path.basename(file_name[0]).split('.')[0]
            targets = targets.view(-1)

            pc = pointcloud.squeeze(0)  # [n_groups, n_points, dims]
            dims = pc.shape[2]

            preds_per_point = {}
            label_per_point = {}

            for v in range(VOTES):
                if v > 0:
                    # new random grouping of the unique points
                    pc, ids = get_sampled_sequence(pc[:N, :], ids[:N], N_POINTS)  # [n_groups, n_points, dims]

                logits, _ = model(pc.transpose(2, 1).to(device))  # [n_groups, n_points, n_classes]
                preds = logits.contiguous().view(-1, N_CLASSES).argmax(1).cpu().numpy()

                pc = pc.reshape(-1, dims)  # [points, dims]

                for i, point_id in enumerate(ids.tolist()):
                    preds_per_point.setdefault(point_id, []).append(int(preds[i]))
                    if v == 0:
                        label_per_point[point_id] = int(targets[i])

            # one row per point (padded duplicates only add votes)
            rows = ['file_name,point_id,label,' + ','.join(f'p{k + 1}' for k in range(VOTES)) + '\n']
            for point_id, point_preds in preds_per_point.items():
                rows.append(f'{file_name},{point_id},{label_per_point[point_id]},'
                            + ','.join(str(p) for p in point_preds[:VOTES]) + '\n')
            with open(os.path.join(output_dir, f'preds-{file_name}.csv'), 'w') as fid:
                fid.writelines(rows)

    print(f'--- TOTAL TIME: {round((time.time() - start_time) / 60, 3)} min ---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='src/results/preds_voting_pn2_01-23-10',
                        help='output directory')
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='src/checkpoints/seg_01-23-10:55_weighted.pth',
                        help='models checkpoint path')
    parser.add_argument('--in_path', type=str, nargs='+', required=True,
                        help='one or more directories of preprocessed .pt windows')
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f'Device: {device}, votes: {VOTES}')

    model = load_model(args.model_checkpoint, N_CLASSES, device)
    os.makedirs(args.output_dir, exist_ok=True)

    for in_path in args.in_path:
        test(args.output_dir, args.n_workers, model, glob.glob(os.path.join(in_path, '*.pt')))
