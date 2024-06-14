import argparse
import os.path
import time
from collections import Counter, defaultdict
import torch
from torch.utils.data import random_split
from src.datasets import CAT3SamplingDataset
import logging
from src.models.pointnet2_sem_seg import PointNet2
from utils.utils import *
from utils.utils_plot import *
from utils.get_metrics import *
from prettytable import PrettyTable
import torch.nn.functional as F
from src.config import *
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from codecarbon import track_emissions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CLASSES = 5
VOTES = 3
N_SAMPLES = 64 # KNN samples
global DATASET

logging.info(f"Device: {device}")
logging.info(f"Votes: {VOTES}")


def load_model(model_checkpoint, n_classes):
    checkpoint = torch.load(model_checkpoint)

    # model
    model = PointNet2(num_classes=n_classes).to(device)
    model = model.apply(weights_init)
    model = model.apply(inplace_relu)

    model.load_state_dict(checkpoint['model'])
    weights = checkpoint['weights']
    batch_size = checkpoint['batch_size']
    learning_rate = checkpoint['lr']
    number_of_points = checkpoint['number_of_points']
    epochs = checkpoint['epoch']
    logging.info('---------- Checkpoint loaded ----------')

    model_name = model_checkpoint.split('/')[-1].split('.')[0].split(':')[0]
    logging.info(f'Model name: {model_name} ')
    logging.info(f"Weights: {weights.cpu().numpy()}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Number of points: {number_of_points}")
    logging.info(f'Model trained for {epochs} epochs')

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name_param, parameter in model.named_parameters():
        # if not parameter.requires_grad: continue
        # parameter.requires_grad = False # Freeze all layers
        params = parameter.numel()
        table.add_row([name_param, params])
        total_params += params
    # print(table)
    logging.info(f"Total Trainable Params: {total_params}")

    return model


# @track_emissions()
def test(output_dir,
         number_of_workers,
         model,
         list_files,
         n_points,
         targets_arr,
         preds_arr):
    # net to eval mode
    model = model.eval()

    # counters
    count_uncertain = 0
    count_sure_pc = 0
    count_no_voting_8k = 0

    # Initialize dataset
    test_dataset = CAT3SamplingDataset(task='segmentation',
                                       number_of_points=n_points,
                                       files=list_files,
                                       return_coord=False,
                                       tile_ids=True)
    # Datalaoders
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)

    preds_tile = torch.zeros((60000000, 5), dtype=torch.int64)
    labels_tile = np.zeros((60000000, 5), dtype=int)

    with torch.no_grad():
        for i, data in enumerate(progressbar(test_dataloader, )):

            pointcloud, targets, file_name, ids, N = data  # [batch, n_samp, n_points, dims]

            ids_ini = ids.squeeze(0).cpu().to(torch.int64)
            # file_name = file_name[0].split('/')[-1].split('.')[0]  # pc_B23_ETehpt421601_w212
            # print(file_name)
            # tile = file_name.split('_')[-2]

            targets = targets.view(-1).cpu()
            targets = torch.nn.functional.one_hot(targets, num_classes=5).numpy()

            pc = pointcloud.squeeze(0).to(device)  # [n_samp, n_points, dims]
            dims = pc.shape[2]
            dupli = False
            knn_smpl = False

            if n_points < pc.shape[0] * pc.shape[1] < 32 * n_points:
                # duplicate points with different samplings
                pc2, ids2 = get_sampled_sequence(pc.view(-1, dims), ids_ini, n_points)
                pc = torch.cat((pc, pc2), dim=0)
                ids = torch.cat((ids_ini, ids2), dim=0)
                dupli = True
            else:
                ids = ids_ini

            for v in range(VOTES):
                if knn_smpl:
                    # get points with higher uncertainty. if all True -> no uncertainty
                    if len(uncertain_ids) <= 1:
                        count_sure_pc += 1
                        break
                    else:
                        count_uncertain += 1
                        # Create a boolean mask with selected_ids
                        mask = torch.isin(ids, torch.FloatTensor(list(uncertain_ids)))  # .to(device)
                        # print(f'Uncertain ids len: {len(uncertain_ids)}')
                        # print(f'Uncertain points shape: {mask.sum()}')

                        # check if less than 25% of points are uncertain to do KNN sampling
                        if 1 < mask.sum() < n_points/4:
                            # pc = pc.to('cpu')  # move to CPU otherwise too much memory in GPU
                            knn_indices = knn_exp_prob(pc[:, :2], mask, n_points=8000, num_samples=N_SAMPLES)

                            # Add a batch dimension to pc
                            pc = torch.unsqueeze(pc, 0)  # .to(device)

                            # Use a loop to concatenate the selected indices
                            selected_indices = [knn_indices[i] for i in range(N_SAMPLES)]
                            pc = torch.cat([pc[:, index, :] for index in selected_indices], dim=0)
                            ids = torch.cat([ids[index] for index in selected_indices], dim=0)  # [tensor]
                        else:
                            # get points with random samplings
                            pc, ids = get_sampled_sequence(pc, ids, n_points)  # [n_samplings, n_points, D]

                elif v == 1:
                    # get points with random samplings
                    pc, ids = get_sampled_sequence(pc, ids, n_points)  # [n_samplings, n_points, D]

                # transpose for model
                pc = pc.transpose(2, 1)

                # get logits from model
                preds, _ = model(pc)  # [batch, n_class, n_points]
                preds = preds.contiguous().view(-1, N_CLASSES)

                # get predictions
                preds = F.softmax(preds, dim=1).data.max(1)[1]
                preds = preds.view(-1).to(dtype=torch.int64)  # [n_points]
                # one hot encoding
                preds = torch.nn.functional.one_hot(preds.cpu(), num_classes=5).to(torch.int64)  # [points, classes]

                pc = pc.transpose(1, 2)
                pc = pc.view(-1, dims)  # [points, dims]

                # plot
                # if v == 2 and len(uncertain_ids) > 10:
                #     preds_all = preds.cpu().numpy()
                #     preds_1 = preds_all[:n_points]
                #     plot_two_pointclouds(pc.cpu().numpy(),
                #                          pc_2=pc[:n_points, :].cpu().numpy(),
                #                          labels=preds_all,
                #                          labels_2=preds_1,
                #                          name=file_name + '_sampled',
                #                          path_plot=os.path.join(output_dir, 'uncertain', 'knn_8K_9s'),
                #                          point_size=2)

                # Update the preds_tile matrix
                preds_tile.scatter_add_(0, ids.unsqueeze(1).expand(-1, 5), preds)
                ids_np = ids.numpy()

                if knn_smpl:
                    break

                if v == 0:
                    # add labels to labels_tile
                    labels_tile[ids_ini.numpy()] = targets

                    # if all point predictions are the same
                    # Calculate the sum of classes across all selected rows
                    total_sum_classes = torch.sum(preds_tile[ids_np], dim=0)
                    all_rows_same_c = (total_sum_classes > 0).sum() == 1

                    # Check if all selected rows have the same class distribution
                    if pc.shape[0] == n_points or all_rows_same_c:
                        count_no_voting_8k += 1
                        break

                    # get unique points and unique ids
                    pc = pc[:N, :]  # [n_points, 8]
                    ids = ids[:N]

                if v == 1 or dupli:
                    # get rows of ids and obtain if there are any votes that are different within those ids
                    # Extract rows corresponding to specific_ids and Calculate the number of unique classes per row
                    unique_classes_per_row = np.count_nonzero(preds_tile[ids_np][:,1:], axis=1)

                    # Find indices of rows with non-unique classes
                    uncertain_ids = set(ids_np[np.where(unique_classes_per_row > 1)[0]])
                    knn_smpl = True

                    if v==1:
                        # get unique points and unique ids
                        pc = pc[:N, :]  # [n_points, 8]
                        ids = ids[:N]

            del preds, pc
            torch.cuda.empty_cache()

    final_class_tile = torch.argmax(preds_tile, dim=1)
    labels_tile = np.argmax(labels_tile, axis=1)

    len_p = len(preds_tile.sum(1) != 0)
    final_class_tile = final_class_tile[:len_p].numpy()
    labels_tile = labels_tile[:len_p]

    # Concatenate with the existing arrays
    preds_arr = np.concatenate((preds_arr, final_class_tile))
    targets_arr = np.concatenate((targets_arr, labels_tile))

    # counters
    print(f'count_no_voting_8k: {count_no_voting_8k}')
    print(f'count_sure_pc: {count_sure_pc}')
    print(f'count_uncertain: {count_uncertain}')

    return targets_arr, preds_arr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='src/results/test_metrics/knn_test_results',
                        help='output directory')
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--n_points', type=int, default=8000, help='number of points per point cloud')
    parser.add_argument('--model_checkpoint', type=str, help='models checkpoint path')
    parser.add_argument('--in_path', type=str)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    # load model
    model = load_model(args.model_checkpoint, N_CLASSES)
    MODEL_NAME = args.model_checkpoint.split('/')[-1].split(':')[0]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # with open(os.path.join(args.output_dir, 'IoU-results-KNN-%s.csv' % MODEL_NAME), 'w+') as fid:
    #     fid.write(
    #         'dataset,n_points,IoU_tower,IoU_lines,IoU_wind_turbine,IoU_ground,IoU_surrounding,IoU_mean,accuracy,time\n')

    for in_path in args.in_path:
        test_files = glob.glob(os.path.join(in_path, '*.pt'))

        start_time = time.time()
        DATASET = in_path.split('/')[-1]
        print(f'DATASET: {DATASET}')

        # list of tiles
        tiles = set([path_f.split('_')[-2].split('.')[0] for path_f in test_files])
        logging.info(f"Number of tiles: {len(tiles)}")

        # Initialize empty arrays
        preds_arr = np.empty(0, dtype=int)
        targets_arr = np.empty(0, dtype=int)

        dataset = 'knn_1V_' + str(N_SAMPLES) + 's_8k' + MODEL_NAME + '_' + DATASET
        print(f"Dataset config: {dataset}")
        print(tiles)

        for tile in tiles:
            print(f'--------- TILE: {tile} --------- ')  # tile='300546' 282553

            test_files = glob.glob(os.path.join(in_path, '*' + tile + '*.pt'))

            targets_arr, preds_arr = test(args.output_dir,
                                          args.n_workers,
                                          model,
                                          test_files,
                                          args.n_points,
                                          targets_arr,
                                          preds_arr
                                          )
        # compute IoUs
        iou = {
            'ground': get_iou_np(targets_arr, preds_arr, 0),
            'tower': get_iou_np(targets_arr, preds_arr, 1),
            'lines': get_iou_np(targets_arr, preds_arr, 2),
            'surr': get_iou_np(targets_arr, preds_arr, 3),
            'wind_turbine': get_iou_np(targets_arr, preds_arr, 4),
        }
        # store IoUs
        iou_arr = [iou['tower'], iou['surr'], iou['wind_turbine'], iou['lines']]  # iou['building'],, iou['ground']
        mean_iou = np.nanmean(iou_arr)
        print('--- GLOBAL METRICS ------------------')
        print('mean iou tower: ', str(round(float(np.nanmean(iou['tower'])) * 100, 3)))
        print('mean iou wires: ', str(round(float(np.nanmean(iou['lines'])) * 100, 3)))
        print('mean iou wind turbine: ', str(round(float(np.nanmean(iou['wind_turbine'])) * 100, 3)))
        print('mean iou ground: ', str(round(float(np.nanmean(iou['ground'])) * 100, 3)))
        print('mean iou surrounding: ', str(round(float(np.nanmean(iou['surr'])) * 100, 3)))
        print('mean iou (no ground): ', str(round(float(mean_iou) * 100, 3)))

        # compute accuracy
        accuracy = None
        # accuracy = accuracy_score(targets_arr, preds_arr)
        # print('accuracy: ', round(float(np.mean(accuracy))*100, 3))
        # print(f'Model trained for {epochs} epochs')

        xstr = lambda x: "None" if x is None else str(round(x, 4) * 100)
        time_min = round((time.time() - start_time) / 60, 3)

        with open(os.path.join(args.output_dir, 'IoU-results-%s.csv' % MODEL_NAME), 'a') as fid:
            fid.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (dataset,
                                                           str(args.n_points),
                                                           xstr(iou['tower']),
                                                           xstr(iou['lines']),
                                                           xstr(iou['wind_turbine']),
                                                           xstr(iou['ground']),
                                                           xstr(iou['surr']),
                                                           xstr(mean_iou),
                                                           accuracy,
                                                           str(time_min)))
        # confusion matrix
        cm = confusion_matrix(targets_arr, preds_arr)
        print(f'confusion matrix: \n{cm}\n')
        with open(os.path.join(args.output_dir, 'confusion_matrix_' + dataset + '.txt'), 'w') as filehandle:
            json.dump(cm.tolist(), filehandle)

        time_min = round((time.time() - start_time) / 60, 3)
        print("--- TOTAL TIME: %s min ---" % str(time_min))
