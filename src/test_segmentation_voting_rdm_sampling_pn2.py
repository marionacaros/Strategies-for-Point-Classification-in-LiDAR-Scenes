import argparse
import gc
import os.path
import time
import torch.cuda
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
# from codecarbon import track_emissions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_CLASSES = 5
VOTES = 13

logging.info(f"Device: {device}")
logging.info(f"Votes: {VOTES}")


# @track_emissions()
def test(output_dir,
         number_of_workers,
         model_checkpoint,
         list_files):
    """
    Iterate VOTES times over samples and store predictions into csv files.

    :param output_dir: directory where csv files are stored
    :param number_of_workers: num cpus
    :param model_checkpoint
    :param list_files
    """

    start_time = time.time()
    checkpoint = torch.load(model_checkpoint)

    # Initialize dataset
    test_dataset = CAT3SamplingDataset(task='segmentation',
                                       number_of_points=8000,
                                       files=list_files,
                                       return_coord=False)

    logging.info(f'Total samples: {len(test_dataset)}')
    logging.info(f'Task: {test_dataset.task}')

    # Datalaoders
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
    number_of_points = checkpoint['number_of_points']
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

    model = model.eval()
    with torch.no_grad():
        for data in progressbar(test_dataloader, ):

            pointcloud, targets, file_name, ids, N = data  # [batch, n_samp,  P, D]
            # last dimension of pointcloud is point ID
            ids = torch.Tensor.int(ids.squeeze(0).cpu())
            # ids=ids[:,0]
            file_name = file_name[0].split('/')[-1].split('.')[0]

            # convert targets to numpy array
            targets = targets.view(-1)

            with open(os.path.join(output_dir, 'preds-%s.csv' % file_name), 'w+') as fid:
                fid.write('file_name,point_id,label,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13\n')

            pc = pointcloud.squeeze(0)  # [n_samp, points, dims]
            dims = pc.shape[2]

            preds_per_point = {}
            label_per_point = {}

            for v in range(VOTES):

                if v > 0:
                    # Create a boolean array indicating whether each value is unique
                    # unique_elements, index = np.unique(pc[:, -1], return_index=True)
                    pc = pc[:N, :]
                    ids = ids[:N]

                    # get tensor with random samplings
                    pc, ids = get_sampled_sequence(pc, ids, number_of_points)  # [n_samplings, n_points, D]

                # transpose for model
                pc = pc.transpose(2, 1)

                # get logits from model
                logits, _ = model(pc[:,:-1,:].to(device))  # [batch, n_points, n_class]
                logits = logits.contiguous().view(-1, N_CLASSES)

                # get predictions
                logits = F.softmax(logits.detach().to('cpu'), dim=1)  # [n_points, classes]
                preds = logits.data.max(1)[1].view(-1).numpy()  # [n_points]

                # get ids
                pc = pc.transpose(2, 1)
                pc = pc.view(-1, dims)  # [points, dims]

                # Store per point predictions with point ids as keys
                for i, point_id in enumerate(ids):
                    if point_id.item() not in preds_per_point.keys():
                        preds_per_point[point_id.item()] = []

                    # append prediction to dict
                    preds_per_point[point_id.item()].append(preds[i])

                    # first round
                    if v == 0: #and point_id not in label_per_point.keys():
                        # write file with point ids and coords
                        # with open(os.path.join(output_dir, 'xyz-%s.csv' % file_name), 'a') as fid:
                        #     fid.write(f'{file_name},{point_id},{x[i]},{y[i]},{z[i]}\n')
                        try:
                            # add target to dict of labels
                            label_per_point[point_id.item()] = targets[i].item()
                        except Exception as e:
                            print(e)

                    # if last round
                    if v == VOTES-1:
                        preds_ = [",{}".format(i) for i in preds_per_point[point_id.item()][:VOTES]]
                        # write point info to file
                        with open(os.path.join(output_dir, 'preds-%s.csv' % file_name), 'a') as fid:
                            fid.write(f'{file_name},{point_id.item()},{label_per_point[point_id.item()]}')
                            fid.writelines(preds_)
                            fid.write('\n')
                    #     winner = Counter(preds_per_point[point_id]).most_common(1)[0][0]
                    #     # Store predictions and targets
                    #     predictions_list.append(winner)
                    #     targets_list.append(label_per_point[point_id])

                del logits

    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                        default='src/results/preds_voting_pn2_01-23-10',
                        help='output directory')
    parser.add_argument('--n_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str,
                        default='src/checkpoints/seg_01-23-10:55_weighted.pth',
                        help='models checkpoint path')
    parser.add_argument('--path_list_files', type=str,
                        default='train_test_files/100x100/test')
    parser.add_argument('--in_path', type=str,
                        default=[
                            '/mnt/QPcotLIDev01/LiDAR/DL_preproc/100x100_s80_p8k_id/test/EMP',
                            '/mnt/QPcotLIDev01/LiDAR/DL_preproc/100x100_s80_p8k_id/test/RIB',
                            '/mnt/QPcotLIDev01/LiDAR/DL_preproc/100x100_s80_p8k_id/test/B29'
                        ])

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    MODEL_NAME = args.model_checkpoint.split('/')[-1].split(':')[0] + '_weighted'

    # test_files = ['test_seg_cat3_files.txt', 'test_seg_rib_files.txt', 'test_seg_b29_files.txt'] #

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for in_path in args.in_path:

        test_files = glob.glob(os.path.join(in_path, '*.pt'))
        # flush GPU memory
        gc.collect()
        torch.cuda.empty_cache()

        test(args.output_dir,
             args.n_workers,
             args.model_checkpoint,
             test_files)

