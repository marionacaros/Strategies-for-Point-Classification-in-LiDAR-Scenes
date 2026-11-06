"""Train PointNet++ on the preprocessed .pt windows (random 80/20 train/val split)."""
import argparse
import datetime
import glob
import logging
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from progressbar import progressbar
from torch.utils.tensorboard import SummaryWriter

from src.config import COLOR_DROPOUT, N_CLASSES, SAMPLES_X_CLASS_CAT3, WEIGHING_METHOD
from src.datasets import CAT3Dataset
from src.models.pointnet2_sem_seg import PointNet2
from utils.get_metrics import EVAL_CLASSES, get_accuracy, get_iou_obj, get_weights4class
from utils.utils import rotate_point_cloud_z, save_checkpoint, weights_init

random.seed(4)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def train(
        path_list_files,
        n_points,
        batch_size,
        epochs,
        learning_rate,
        number_of_workers,
        model_checkpoint,
        run_name='',
        class_weights=None,
        n_train_points=4096):
    start_time = time.time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Device: {device}')

    # Tensorboard location and run name
    location = 'src/runs/prod/'
    NAME = 'seg_' + datetime.datetime.now().strftime("%m-%d-%H:%M") + (('_' + run_name) if run_name else '')
    writer_train = SummaryWriter(location + NAME + '_train')
    writer_val = SummaryWriter(location + NAME + '_val')
    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    train_files = glob.glob(os.path.join(path_list_files, '*pt'))
    random.shuffle(train_files)
    lim = round(0.8 * len(train_files))
    val_files = train_files[lim:]
    train_files = train_files[:lim]
    logging.info(f'Train windows: {len(train_files)}, val windows: {len(val_files)}')

    train_dataloader = torch.utils.data.DataLoader(CAT3Dataset(number_of_points=n_points, files=train_files),
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers,
                                                   drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(CAT3Dataset(number_of_points=n_points, files=val_files),
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=number_of_workers,
                                                 drop_last=True)
    pointnet = PointNet2(num_classes=N_CLASSES).to(device)
    pointnet = pointnet.apply(weights_init)
    logging.info(f"Total Trainable Params: {sum(p.numel() for p in pointnet.parameters() if p.requires_grad)}")

    optimizer = optim.AdamW(pointnet.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                            eps=1e-08, weight_decay=1e-4)

    # class weights for the loss: explicit --class_weights, else WEIGHING_METHOD from src/config.py
    if class_weights:
        c_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    elif WEIGHING_METHOD:
        c_weights = get_weights4class(WEIGHING_METHOD,
                                      n_classes=N_CLASSES,
                                      samples_per_cls=SAMPLES_X_CLASS_CAT3,
                                      beta=0.999).to(device)
    else:
        c_weights = None
    logging.info(f'Class weights: {c_weights}')

    LS = 0.0  # label smoothing
    ce_loss = torch.nn.CrossEntropyLoss(weight=c_weights, reduction='mean', ignore_index=-1, label_smoothing=LS)

    if model_checkpoint:
        logging.info(f'Resuming from {model_checkpoint}')
        checkpoint = torch.load(model_checkpoint, map_location=device)
        pointnet.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.param_groups[0]['lr'] = learning_rate

    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_vloss = 1_000_000.
    epochs_since_improvement = 0

    for epoch in progressbar(range(epochs)):
        loss = {'train': [], 'val': []}
        acc = {'train': [], 'val': []}
        # per-batch IoU of each evaluated class (ground is not evaluated)
        iou = {split: {name: [] for name in EVAL_CLASSES.values()} for split in ('train', 'val')}

        def log_batch(split, metrics, targets, preds):
            targets = targets.view(-1)
            preds = preds.view(-1)
            metrics = get_accuracy(preds, targets, metrics)
            for label, name in EVAL_CLASSES.items():
                iou[split][name].append(get_iou_obj(targets, preds, label))
            loss[split].append(metrics['loss'].cpu().item())
            acc[split].append(metrics['accuracy'])

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:
            log_batch('train', *train_loop(data, optimizer, ce_loss, pointnet, True, device=device,
                                           n_train_points=n_train_points))
        scheduler_cosine.step()

        # --------------------------------------------- val loop -----------------------------------------------
        with torch.no_grad():
            for data in val_dataloader:
                log_batch('val', *train_loop(data, optimizer, ce_loss, pointnet, False, device=device,
                                             n_train_points=n_train_points))

        # Save checkpoint on best validation loss
        if np.mean(loss['val']) < best_vloss:
            save_checkpoint(NAME, epoch, epochs_since_improvement, pointnet, optimizer, acc['val'][-1],
                            batch_size, learning_rate, n_points, weights=c_weights, label_smoothing=LS,
                            color_dropout=COLOR_DROPOUT)
            logging.info(f'model {NAME} saved at epoch {epoch}')
            epochs_since_improvement = 0
            best_vloss = np.mean(loss['val'])
        else:
            epochs_since_improvement += 1

        # Tensorboard
        for split, writer in (('train', writer_train), ('val', writer_val)):
            writer.add_scalar('loss', np.nanmean(loss[split]), epoch)
            writer.add_scalar('accuracy', np.nanmean(acc[split]), epoch)
            for name, values in iou[split].items():
                writer.add_scalar(f'_iou_{name}', np.nanmean(values), epoch)
            writer.flush()
        writer_val.add_scalar('epochs_since_improvement', epochs_since_improvement, epoch)
        writer_val.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    logging.info(f'--- TOTAL TIME: {round((time.time() - start_time) / 3600, 3)} h ---')


def train_loop(data, optimizer, ce_loss, pointnet, train=True, device='cuda', n_train_points=4096):
    """
    One training (or validation) step on a batch of windows.

    :return: metrics ({'loss'}), targets [batch * n_train_points], preds [batch * n_train_points]
    """
    metrics = {}
    pc, targets, _ = data
    pc = pc.data.numpy()

    # random subset of points (without repetition)
    n_available = pc.shape[1]
    random_indices = random.sample(range(n_available), min(n_train_points, n_available))
    pc = pc[:, random_indices, :]
    targets = targets[:, random_indices]

    # rotation around z (rotates the x, y channels)
    pc[:, :, :3] = rotate_point_cloud_z(pc[:, :, :3])
    pc = torch.Tensor(pc)

    pc, targets = pc.to(device), targets.to(device)  # [batch, n_samples, dims], [batch, n_samples]
    pc = pc.transpose(2, 1)  # [batch, dims, n_samples]

    # color dropout: zero G, B and NDVI
    if train and random.random() < COLOR_DROPOUT:
        pc[:, 4:, :] = 0

    optimizer.zero_grad()
    pointnet.train(train)

    logits, _ = pointnet(pc)
    logits = logits.contiguous().view(-1, N_CLASSES)
    targets = targets.contiguous().view(-1)

    metrics['loss'] = ce_loss(logits, targets)
    preds = logits.detach().argmax(1).cpu()

    if train:
        metrics['loss'].backward()
        optimizer.step()

    return metrics, targets.detach().cpu(), preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_list_files', type=str, required=True, help='directory of .pt windows')
    parser.add_argument('--number_of_points', type=int, default=8000, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--number_of_workers', type=int, default=4, help='number of workers for the dataloader')
    parser.add_argument('--n_train_points', type=int, default=4096,
                        help='points randomly drawn from each window at every step')
    parser.add_argument('--model_checkpoint', type=str, default=None,
                        help='checkpoint to resume from (omit to train from scratch)')
    parser.add_argument('--run_name', type=str, default='',
                        help='suffix of the run name used for the checkpoint and TensorBoard logs')
    parser.add_argument('--class_weights', type=float, nargs=N_CLASSES, default=None,
                        help='CE loss weight per class, e.g. the ones stored in seg_01-23-10:55_weighted.pth: '
                             '0.1718 0.2430 0.2430 0.0992 0.2430')

    args = parser.parse_args()
    train(
        args.path_list_files,
        args.number_of_points,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.number_of_workers,
        args.model_checkpoint,
        run_name=args.run_name,
        class_weights=args.class_weights,
        n_train_points=args.n_train_points)
