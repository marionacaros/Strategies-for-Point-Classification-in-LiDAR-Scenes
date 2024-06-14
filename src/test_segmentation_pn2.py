import argparse
import time
from torch.utils.data import random_split
from src.datasets import CAT3SamplingDataset
import logging
from src.models.pointnet2_sem_seg import PointNet2
# from src.models.pointnet2_2k import PointNet2
from src.prod.train_segmentation_pn2 import weights_init, inplace_relu
from utils.utils import *
from utils.utils_plot import *
from utils.get_metrics import *
from prettytable import PrettyTable
import torch.nn.functional as F
from src.config import *
from sklearn.metrics import confusion_matrix, accuracy_score
import json
import gc
# from codecarbon import track_emissions

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

N_CLASSES = 5
VOTES = 1
MODEL_NAME = ''

logging.info(f"Device: {device}")
logging.info(f"Votes: {VOTES}")


# @track_emissions()
def test(output_dir,
         number_of_workers,
         model_checkpoint,
         list_files,
         dataset,
         n_points):
    start_time = time.time()
    checkpoint = torch.load(model_checkpoint)
    # classes = ['other', 'tower', 'lines', 'veg', 'wind_turbine']

    # Initialize dataset
    test_dataset = CAT3SamplingDataset(task='segmentation',
                                       number_of_points=n_points,
                                       files=list_files,
                                       return_coord=False)

    logging.info(f'Total samples: {len(test_dataset)}')
    logging.info(f'Task: {test_dataset.task}')

    # Dataloaders
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)
    # models
    model = PointNet2(num_classes=N_CLASSES).to(device)
    model = model.apply(weights_init)
    model = model.apply(inplace_relu)

    logging.info('--- Checkpoint loaded ---')
    model.load_state_dict(checkpoint['model'])
    weights = checkpoint['weights']
    batch_size = checkpoint['batch_size']
    learning_rate = checkpoint['lr']
    number_of_points = n_points
    epochs = checkpoint['epoch']

    logging.info(f"Weights: {weights}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Learning rate: {learning_rate}")
    logging.info(f"Number of points: {number_of_points}")
    logging.info(f'Model trained for {epochs} epochs')
    model_name = model_checkpoint.split('/')[-1].split('.')[0]
    logging.info(f'Model name: {model_name} ')

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name_param, parameter in model.named_parameters():
        # if not parameter.requires_grad: continue
        # parameter.requires_grad = False # Freeze all layers
        params = parameter.numel()
        table.add_row([name_param, params])
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")

    # Initialize empty tensors
    preds_tensor_gpu = torch.empty(0, dtype=torch.long, device=device)
    targets_tensor = torch.empty(0, dtype=torch.long)
    # Initialize empty tensors on CPU
    preds_tensor_cpu = torch.empty(0, dtype=torch.long)

    model = model.eval()
    with torch.no_grad():
        for data in progressbar(test_dataloader, ):
            pointcloud, targets, file_name = data  # [batch, n_samp, points, dims]
            dims = pointcloud.shape[3]  # last dimension of pointcloud is point ID
            # file_name = file_name[0].split('/')[-1].split('.')[0]

            # transpose for model
            pc_t = pointcloud.squeeze(0).transpose(2, 1)  # [n_samp, dims, points]

            logits, _ = model(pc_t[:, :-1, :].to(device))  # [batch, n_points, n_class]
            logits = logits.contiguous().view(-1, N_CLASSES)

            pointcloud = pointcloud.contiguous().view(-1, dims)
            # Create a boolean array indicating whether each value is unique
            unique_elements, index = np.unique(pointcloud[:, -1], return_index=True)

            # get predictions
            probs = F.softmax(logits, dim=1)  # [n_points, classes]
            preds = probs.data.max(1)[1].view(-1).to(dtype=torch.int)

            # append for IoU
            preds_tensor_gpu = torch.cat((preds_tensor_gpu, preds[index]), dim=0)
            targets_tensor = torch.cat((targets_tensor, targets.view(-1)[index]), dim=0)

            del logits

            # Move tensors to CPU after processing X samples
            if len(preds_tensor_gpu) >= 100:
                preds_tensor_cpu = torch.cat((preds_tensor_cpu, preds_tensor_gpu.cpu()), dim=0)

                # Reset GPU tensors
                preds_tensor_gpu = torch.empty(0, dtype=torch.long, device=device)
                torch.cuda.empty_cache()

    # At the end of the loop, move any remaining tensors on GPU to CPU
    preds_tensor_cpu = torch.cat((preds_tensor_cpu, preds_tensor_gpu.cpu()), dim=0)

    predictions_array = preds_tensor_cpu.numpy()
    targets_array = targets_tensor.numpy()

    # compute IoUs
    iou = {
        'other': get_iou_np(targets_array, predictions_array, 0),
        'tower': get_iou_np(targets_array, predictions_array, 1),
        'lines': get_iou_np(targets_array, predictions_array, 2),
        'veg': get_iou_np(targets_array, predictions_array, 3),
        'wind_turbine': get_iou_np(targets_array, predictions_array, 4),
    }
    # store IoUs
    iou_arr = [iou['tower'], iou['veg'], iou['wind_turbine'], iou['lines'], iou['other']]  # iou['building'],
    mean_iou = np.nanmean(iou_arr)

    # compute accuracy
    accuracy = accuracy_score(targets_array, predictions_array)

    # confusion matrix
    cm = confusion_matrix(targets_array, predictions_array)
    print(f'confusion matrix: {cm}\n')

    with open(os.path.join(output_dir, 'confusion_matrix_' + dataset + '.txt'), 'w') as filehandle:
        json.dump(cm.tolist(), filehandle)

    xstr = lambda x: "None" if x is None else str(round(x, 3))

    print('-------------')
    print('mean_iou_ground: ', round(float(np.nanmean(iou['other'])), 3))
    print('mean_iou_tower: ', round(float(np.nanmean(iou['tower'])), 3))
    print('mean_iou_lines: ', round(float(np.nanmean(iou['lines'])), 3))
    print('mean_iou_veg: ', round(float(np.nanmean(iou['veg'])), 3))
    print('mean_iou_wind_turbine: ', xstr(np.nanmean(iou['wind_turbine'])))
    print('mean_iou: ', round(float(mean_iou), 3))
    print('accuracy: ', round(float(accuracy), 3))
    print(f'Model trained for {epochs} epochs')
    time_min = round((time.time() - start_time) / 60, 3)
    print("--- TOTAL TIME: %s min ---" % str(time_min))

    with open(os.path.join(output_dir, 'IoU-results-%s.csv' % MODEL_NAME), 'a') as fid:
        fid.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (dataset,
                                                       str(number_of_points),
                                                       xstr(iou['veg']),
                                                       xstr(iou['tower']),
                                                       xstr(iou['lines']),
                                                       xstr(iou['other']),
                                                       xstr(iou['wind_turbine']),
                                                       xstr(mean_iou),
                                                       accuracy,
                                                       str(time_min)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='src/results/test_metrics',
                        help='output directory')
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--n_points', type=int, default=4000, help='number of points to sample from original point cloud')
    parser.add_argument('--model_checkpoint', type=str,
                        default='src/checkpoints/seg_01-23-10:55_weighted.pth',
                        help='models checkpoint path')
    parser.add_argument('--path_list_files', type=str)
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    MODEL_NAME = args.model_checkpoint.split('/')[-1].split(':')[0] + '_w'

    test_files = ['test_seg_cat3_files.txt', 'test_seg_rib_files.txt', 'test_seg_b29_files.txt']

    for test_f in test_files:
        # get test files
        with open(os.path.join(args.path_list_files, test_f), 'r') as f:
            test_files = f.read().splitlines()

        dataset = test_f.split('_')[2] + '_' + args.model_checkpoint.split('/')[-1]
        logging.info(f"Dataset: {dataset}")

        gc.collect()
        torch.cuda.empty_cache()

        test(args.output_dir,
             args.n_workers,
             args.model_checkpoint,
             test_files,
             dataset,
             args.n_points)
