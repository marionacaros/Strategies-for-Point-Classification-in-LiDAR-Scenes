import argparse
import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
from src.datasets import CAT3Dataset
from src.models.pointnet2_sem_seg import *
from utils.utils import *
from utils.get_metrics import *
import logging
import datetime
from prettytable import PrettyTable
import random
#from utils.utils_plot import plot_pc_tensorboard
from src.config import *
import sys

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
        model_checkpoint):
    start_time = time.time()

    if torch.cuda.is_available():
        logging.info(f"cuda available")
        device = 'cuda'
    else:
        logging.info(f"cuda not available")
        device = 'cpu'

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'src/runs/prod/'
    NAME = 'seg_' + now.strftime("%m-%d-%H:%M") + '_buildCOSTA20w'

    writer_train = SummaryWriter(location + NAME + '_train')
    writer_val = SummaryWriter(location + NAME + '_val')
    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    train_files = glob.glob(os.path.join(path_list_files, '*pt'))

    random.shuffle(train_files)
    lim = round(0.8 * len(train_files))
    val_files = train_files[lim:]
    train_files = train_files[:lim]

    print(f'len train files: {len(train_files)}')
    print(f'len val files: {len(val_files)}')

    # Initialize datasets
    train_dataset = CostaDataset(
        task='segmentation',
        number_of_points=n_points,
        files=train_files,
        fixed_num_points=True)
    val_dataset = CostaDataset(
        task='segmentation',
        number_of_points=n_points,
        files=val_files,
        fixed_num_points=True)

    logging.info(f'Samples for training: {len(train_dataset)}')
    logging.info(f'Samples for validation: {len(val_dataset)}')
    logging.info(f'Task: {train_dataset.task}')

    # Datalaoders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers,
                                                   drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=number_of_workers,
                                                 drop_last=True)
    # models
    pointnet = PointNet2(num_classes=N_CLASSES)
    pointnet.to(device)
    pointnet = pointnet.apply(weights_init)
    pointnet = pointnet.apply(inplace_relu)

    # print models and parameters
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in pointnet.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    logging.info(f"Total Trainable Params: {total_params}")

    # optimizer AdamW
    optimizer = optim.AdamW(pointnet.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                            eps=1e-08, weight_decay=1e-4)
    # optimizer = optim.Adam(pointnet.parameters(), lr=learning_rate, betas=(0.9, 0.999),
    #                        eps=1e-08,
    #                        weight_decay=1e-4)

    c_weights = None

    # samples x class for weight computation
    # c_weights = get_weights4class(WEIGHING_METHOD,
    #                               n_classes=N_CLASSES,
    #                               samples_per_cls=SAMPLES_X_CLASS_CAT3,
    #                               beta=0.999).to(device)
    print(f'Weights: {c_weights}')

    # loss
    # LS = 0.01
    LS = 0.0
    ce_loss = torch.nn.CrossEntropyLoss(weight=c_weights, reduction='mean', ignore_index=-1, label_smoothing=LS)

    if model_checkpoint:
        print('Loading checkpoint')
        checkpoint = torch.load(model_checkpoint)
        pointnet.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.param_groups[0]['lr'] = learning_rate

    # schedulers
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, 80)  # 80
    # scheduler_multistep = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                  milestones=[10, 20, 100, 200],  # List of epoch indices
    #                                  gamma=0.1)  # Multiplicative factor of learning rate decay

    best_vloss = 1_000_000.
    epochs_since_improvement = 0

    for epoch in progressbar(range(epochs)):
        epoch_train_loss = []
        epoch_train_acc = []
        epoch_val_loss = []
        epoch_val_acc = []
        iou = {
            'tower_train': [],
            'tower_val': [],
            'veg_train': [],
            'veg_val': [],
            'ground_train': [],
            'ground_val': [],
            'wires_train': [],
            'wires_val': [],
            'turbine_train': [],
            'turbine_val': [],
            'mean_iou_train': [],
            'mean_iou_val': [],
        }

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:
            metrics, targets, preds = train_loop(data, optimizer, ce_loss, pointnet, writer_train,
                                                 True, epoch, device=device)
            targets = targets.view(-1)
            preds = preds.view(-1)
            # compute metrics
            metrics = get_accuracy(preds, targets, metrics)

            # Segmentation labels:
            # 0 -> other classes we're not interested and rest of ground
            # 1 -> tower
            # 2 -> lines
            # 3 -> vegetation
            # 4 -> wind_turbine
            iou['ground_train'].append(get_iou_obj(targets, preds, 0))
            iou['tower_train'].append(get_iou_obj(targets, preds, 1))
            iou['wires_train'].append(get_iou_obj(targets, preds, 2))
            iou['veg_train'].append(get_iou_obj(targets, preds, 3))
            iou['turbine_train'].append(get_iou_obj(targets, preds, 4))

            # tensorboard
            epoch_train_loss.append(metrics['loss'].cpu().item())
            epoch_train_acc.append(metrics['accuracy'])

        scheduler_cosine.step()
        # --------------------------------------------- val loop ---------------------------------------------

        with torch.no_grad():

            for data in val_dataloader:
                metrics, targets, preds = train_loop(data, optimizer, ce_loss, pointnet, writer_val,
                                                     False, 1, device=device)
                targets = targets.view(-1)
                preds = preds.view(-1)
                metrics = get_accuracy(preds, targets, metrics)

                iou['ground_val'].append(get_iou_obj(targets, preds, 0))
                iou['tower_val'].append(get_iou_obj(targets, preds, 1))
                iou['wires_val'].append(get_iou_obj(targets, preds, 2))
                iou['veg_val'].append(get_iou_obj(targets, preds, 3))
                iou['turbine_val'].append(get_iou_obj(targets, preds, 4))

                # tensorboard
                epoch_val_loss.append(metrics['loss'].cpu().item())  # in val ce_loss and total_loss are the same
                epoch_val_acc.append(metrics['accuracy'])

        # ------------------------------------------------------------------------------------------------------
        # Save checkpoint
        if np.mean(epoch_val_loss) < best_vloss:
            name = NAME
            save_checkpoint(name, epoch, epochs_since_improvement, pointnet, optimizer, metrics['accuracy'],
                            batch_size, learning_rate, n_points, weights=c_weights, label_smoothing=LS,
                            color_dropout=COLOR_DROPOUT)
            print(f'model {name} saved at epoch {epoch}')
            epochs_since_improvement = 0
            best_vloss = np.mean(epoch_val_loss)

        else:
            epochs_since_improvement += 1
        # if epochs_since_improvement > 50:
        #     exit()

        # Tensorboard
        writer_train.add_scalar('loss', np.nanmean(epoch_train_loss), epoch)
        writer_val.add_scalar('loss', np.nanmean(epoch_val_loss), epoch)
        writer_train.add_scalar('accuracy', np.nanmean(epoch_train_acc), epoch)
        writer_val.add_scalar('accuracy', np.nanmean(epoch_val_acc), epoch)
        writer_val.add_scalar('epochs_since_improvement', epochs_since_improvement, epoch)
        writer_val.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer_train.add_scalar('_iou_tower', np.nanmean(iou['tower_train']), epoch)
        writer_val.add_scalar('_iou_tower', np.nanmean(iou['tower_val']), epoch)
        writer_train.add_scalar('_iou_ground', np.nanmean(iou['ground_train']), epoch)
        writer_val.add_scalar('_iou_ground', np.nanmean(iou['ground_val']), epoch)
        writer_train.add_scalar('_iou_veg', np.nanmean(iou['veg_train']), epoch)
        writer_val.add_scalar('_iou_veg', np.nanmean(iou['veg_val']), epoch)
        writer_train.add_scalar('_iou_windturbine', np.nanmean(iou['turbine_train']), epoch)
        writer_val.add_scalar('_iou_windturbine', np.nanmean(iou['turbine_val']), epoch)
        writer_train.flush()
        writer_val.flush()

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))


def train_loop(data, optimizer, ce_loss, pointnet, w_tensorboard=None, train=True,
               epoch=0, device='cuda'):
    """
    :return:
    metrics, targets, preds, last_epoch
    """
    metrics = {'accuracy': []}
    pc, targets, filenames = data
    pc = pc.data.numpy()
    fname_0 = filenames[0].split('/')[-1].split('.')[0]

    # Sample. Generate random indices without repetition
    random_indices = random.sample(range(0, 8000), 4096)
    pc = pc[:, random_indices, :]
    targets = targets[:, random_indices]

    # point cloud rotation
    pc[:, :, :3] = rotate_point_cloud_z(pc[:, :, :3])
    pc = torch.Tensor(pc)

    pc, targets = pc.to(device), targets.to(device)  # [batch, n_samples, dims], [batch, n_samples]
    pc = pc.transpose(2, 1)  # [16,5,4096]

    # color dropout -> Fill the specified positions with zeros
    if train == True:
        if random.random() < COLOR_DROPOUT:
            pc[:, 4:, :] = 0

    # Pytorch accumulates gradients. We need to clear them out before each instance
    optimizer.zero_grad()
    if train:
        pointnet = pointnet.train()
    else:
        pointnet = pointnet.eval()

    # PointNet model
    logits, feat_transform = pointnet(pc)
    logits = logits.contiguous().view(-1, N_CLASSES)
    targets = targets.contiguous().view(-1)

    # CrossEntropy loss
    metrics['loss'] = ce_loss(logits, targets)
    targets_pc = targets.detach().cpu()

    # get predictions
    probs = F.log_softmax(logits.detach().to('cpu'), dim=1)
    preds = torch.LongTensor(probs.data.max(1)[1])

    # plot predictions in Tensorboard
    # if epoch % 10 == 0 and random.random() < 0.05 and 4 in set(targets_pc[:4096].numpy()):
    #     preds_plot, targets_plot = preds[:4096], targets_pc[:4096]
        # Tensorboard
    #     plot_pc_tensorboard(pc[0, :, :].T.cpu(), targets_plot, None, fname_0 +'_targets', step=epoch,
    #                         classes=N_CLASSES)
    #     plot_pc_tensorboard(pc[0, :, :].T.cpu(), preds_plot, None, fname_0 + '_predictions', step=epoch,
    #                         classes=N_CLASSES)

    if train:
        metrics['loss'].backward()
        optimizer.step()

    return metrics, targets_pc, preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_list_files', type=str)
    parser.add_argument('--number_of_points', type=int, default=8000, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--number_of_workers', type=int, default=4, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str,
                        default='src/checkpoints/seg_01-22-12:43_weighted.pth',
                        help='models checkpoint path')

    args = parser.parse_args()
    train(
        args.path_list_files,
        args.number_of_points,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.number_of_workers,
        args.model_checkpoint)
        
