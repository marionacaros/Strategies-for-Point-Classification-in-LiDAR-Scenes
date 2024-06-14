import argparse
from tqdm import tqdm
from utils.utils import *
import time
import random
import multiprocessing
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

global OUT_PATH, W_SIZE, MAX_Z, MAX_I, STORE_DOUBLE, DATASET_NAME
NUM_CPUS = 16


def preprocessing(pc):
    """
    Perform preprocessing on a given point cloud.

    Steps:
    1. Remove outliers and points with negative z-coordinates.
    2. Calculate and clip the Normalized Difference Vegetation Index (NDVI) within the range [-1, 1].
    3. Check the number of non-ground points and augment the dataset by adding more ground points if needed.
    4. If there are already enough non-ground points, remove the ground points.

    Parameters:
    - pc (numpy.ndarray): Input point cloud data.

    Returns:
    - pc numpy.ndarray: Processed point cloud after the specified preprocessing steps.
    """
    # Remove outliers (points above max_z)
    pc = pc[pc[:, 10] <= MAX_Z]
    # Remove points z < 0
    pc = pc[pc[:, 10] >= 0]

    # add NDVI
    pc[:, 9] = get_ndvi(pc[:, 8], pc[:, 5])  # range [-1, 1]
    pc[:, 9] = np.clip(pc[:, 9], -1, 1.0)

    # Check if points different from ground < n_points
    len_pc = pc[pc[:, 3] != 2].shape[0]
    if 100 < len_pc < N_POINTS:
        len_needed_p = N_POINTS - len_pc

        # Get indices of ground points
        labels = pc[:, 3]
        # i_terrain = [i for i in range(len(labels)) if labels[i] == 2.0]
        i_terrain = np.where(labels == 2.0)[0]

        # if we have enough points of ground to cover missed points
        if len_needed_p < len(i_terrain):
            needed_i = random.sample(list(i_terrain), k=len_needed_p)
        else:
            needed_i = i_terrain

        # store points needed
        points_needed_terrain = pc[needed_i, :]

        # remove terrain points
        pc = pc[pc[:, 3] != 2, :]

        # store only needed terrain points
        pc = np.concatenate((pc, points_needed_terrain), axis=0)

    # if enough points, remove ground
    elif len_pc >= N_POINTS:
        pc = pc[pc[:, 3] != 2, :]

    return pc


def parallel_proc(files_list, num_cpus):
    p = multiprocessing.Pool(processes=num_cpus)
    # Use tqdm with imap_unordered
    with tqdm(total=len(files_list)) as pbar:
        for _ in p.imap_unordered(split_pointcloud, files_list):
            pbar.update(1)  # Update progress bar for each completed task
    p.close()
    p.join()


def get_ndvi(nir, red):
    a = (nir - red)
    b = (nir + red)
    c = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    return c


def split_pointcloud(f):
    """
    1 - Remove ground (categories 2, 8, 13)
    2 - Split point cloud into windows of size W_SIZE.
    3 - Add HAG and NDVI

    :param f: file path
    """

    f_name = f.split('/')[-1].split('.')[0] 
    data_f = laspy.read(f)

    # check file is not empty
    if len(data_f.x) > 0:
        if DATASET_NAME != 'EMP':
            try:
                pc = np.vstack((data_f.x,
                                data_f.y,
                                data_f.z,
                                data_f.classification,  # 3
                                data_f.intensity / MAX_I,  # 4
                                data_f.red / 65536.0,  # 5
                                data_f.green / 65536.0,  # 6
                                data_f.blue / 65536.0,  # 7
                                data_f.nir / 65536.0,  # 8
                                np.zeros(len(data_f.x)),  # 9  NDVI
                                data_f.Distance / (1000 * MAX_Z),  # 10  HAG
                                np.arange(len(data_f.x))  # 11 point ID
                                ))
            except AttributeError as e:
                print(e)
                print(f)
                return
        else:
            # Alt Emporda data
            try:
                pc = np.vstack((data_f.x,
                                data_f.y,
                                data_f.z,
                                data_f.classification,  # 3
                                data_f.intensity / MAX_I,  # 4
                                data_f.red / 65536.0,  # 5
                                data_f.green / 65536.0,  # 6
                                data_f.blue / 65536.0,  # 7
                                data_f.nir / 65536.0,  # 8
                                ))
                data_f = laspy.read('/dades/LIDAR/towers_detection/LAS_CAT3_HAG/' + f_name + '.las')
                pc2 = np.vstack((np.zeros(len(data_f.x)),  # 9  NDVI
                                 data_f.HeightAboveGround / MAX_Z,  # 10  HAG
                                 np.arange(len(data_f.x))))  # 11 point ID

                pc = np.concatenate((pc, pc2), axis=0)
            except AttributeError as e:
                print(e)
                print(f)
                return

        pc = pc.transpose()
        # Remove all categories of ground points
        pc = pc[pc[:, 3] != 7.]  # negative points
        pc = pc[pc[:, 3] != 11.]  # air points
        pc = pc[pc[:, 3] != 24.]  # overlap points
        # pc = pc[pc[:, 3] != 13.]  # other ground points

        # Remove sensor noise
        pc = pc[pc[:, 3] != 30.]
        pc = pc[pc[:, 3] != 31.]
        pc = pc[pc[:, 3] != 99.]
        pc = pc[pc[:, 3] != 102.]
        pc = pc[pc[:, 3] != 103.]
        pc = pc[pc[:, 3] != 104.]
        pc = pc[pc[:, 3] != 105.]
        pc = pc[pc[:, 3] != 106.]
        pc = pc[pc[:, 3] != 135.]

        # key DEM points to ground
        indices = np.where(pc[:, 3] == 8.)
        pc[indices, 3] = 2.

        # if B29 change category of wind turbine from other towers
        if DATASET_NAME == 'B29':
            indices = np.where(pc[:, 3] == 18.)
            pc[indices, 3] = 29.

        pc = pc.transpose()

        i_w = 0
        x_min, y_min, z_min = pc[0].min(), pc[1].min(), pc[2].min()
        x_max, y_max, z_max = pc[0].max(), pc[1].max(), pc[2].max()

        for y in range(round(y_min), round(y_max), STRIDE):
            bool_w_y = np.logical_and(pc[1] < (y + W_SIZE), pc[1] > y)

            for x in range(round(x_min), round(x_max), STRIDE):
                bool_w_x = np.logical_and(pc[0] < (x + W_SIZE), pc[0] > x)
                bool_w = np.logical_and(bool_w_x, bool_w_y)
                i_w += 1

                if any(bool_w):
                    if pc[:, bool_w].shape[1] >= N_POINTS:  # check size (ground + other)
                        pc_w = pc[:, bool_w]
                        pc_w = pc_w.transpose()

                        # store torch file
                        fileName = 'pc_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w)

                        # IF TRAINING DATA -> set name for classification
                        unique, counts = np.unique(pc_w[:, 3].astype(int), return_counts=True)
                        dic_counts = dict(zip(unique, counts))
                        if 35 in dic_counts.keys():
                            fileName = 'crane_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w)
                        elif 29 in dic_counts.keys():
                            if dic_counts[29] >= 10:  # check that number of points is >= 10
                                fileName = 'windturbine_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w)
                        elif 15 in dic_counts.keys():
                            if dic_counts[15] >= 10:  # check that number of points is >= 10
                                fileName = 'tower_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w)
                        elif 14 in dic_counts.keys():
                            if dic_counts[14] >= 10:
                                fileName = 'lines_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w)

                        # Check how many points are not ground
                        len_pc = pc_w[pc_w[:, 3] != 2].shape[0]
                        if len_pc > 1000:

                            pc_w = preprocessing(pc_w)  # [points, 11]
                            if pc_w.shape[0] >= N_POINTS:
                                out_fileName = os.path.join(OUT_PATH, fileName)

                                if STORE_DOUBLE:
                                    torch.save(torch.DoubleTensor(pc_w), out_fileName + '.pt')
                                else:
                                    torch.save(torch.FloatTensor(pc_w), out_fileName + '.pt')


if __name__ == '__main__':

    """
    Last version of preporcessing
    All LAS files must have Distance attribute
    1 - Remove ground (categories 2, 8) if size > n_points (i.e. 8000)
    2 - Split point cloud into windows of size W_SIZE. (i.e. 100 x 100)
    3 - Add HAG and NDVI
    
    Output is tensor containing: x,y,z,I,NDVI,HAG,constr_sampl
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path')
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--n_points', type=int, default=8000)
    parser.add_argument('--w_size', type=int, default=100)
    parser.add_argument('--stride', default=50)
    parser.add_argument('--max_z', type=float, default=200.0)
    parser.add_argument('--max_intensity', type=float, default=5000.0)

    args = parser.parse_args()
    start_time = time.time()

    DATASET_NAME = 'EMP'
    STORE_DOUBLE = True

    OUT_PATH = args.out_path
    N_POINTS = args.n_points
    W_SIZE = args.w_size
    MAX_Z = args.max_z
    MAX_I = args.max_intensity
    STRIDE = args.stride

    # check if path is directory
    if os.path.isdir(args.in_path):
        files = glob.glob(os.path.join(args.in_path, '*.las'))

    # check if path is a file
    elif os.path.isfile(args.in_path):
        with open(args.in_path, 'r') as f:
            files = f.read().splitlines()
    else:
        print("Not a valid path")

    logging.info(f'Stride: {STRIDE}')
    logging.info(f'Output path: {args.out_path}')
    logging.info(f'Num of files: {len(files)}')

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # Multiprocessing
    parallel_proc(files, num_cpus=NUM_CPUS)
    # Uncomment to avoid multiprocessing
    # for file in tqdm(files):
    #     split_pointcloud(file)

    print("--- Dataset preprocessing time: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
