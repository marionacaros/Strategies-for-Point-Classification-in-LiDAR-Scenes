import glob
import numpy as np
import torch
from progressbar import progressbar
import os
import pickle
import laspy


def tensor2csv(t, filename):
    for i in range(len(t)):
        with open(os.path.join(filename + '.csv'), 'a') as fid:
            fid.write(f'{t[i, 0].item()}, {t[i, 1].item()}, {t[i, 2].item()}, {t[i, 3].item()}, '
                      f'{t[i, 4].item()}, {t[i, 5].item()}, {t[i, 6].item()}, {t[i, 7].item()} \n')


def tensor2las(pc, filename, path):
    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.4")
    # header.add_extra_dim(laspy.ExtraBytesParams(name="ndvi", type=np.int32))
    # header.add_extra_dim(laspy.ExtraBytesParams(name="hag", type=np.int32))
    # header.add_extra_dim(laspy.ExtraBytesParams(name="constr_flag", type=np.int8))
    # header.offsets = np.min(pc, axis=0)
    # header.scales = np.array([1, 1, 1])

    # 2. Create a Las
    las = laspy.LasData(header)

    las.x = pc[:, 0]
    las.y = pc[:, 1]
    las.z = pc[:, 2]
    las.classification = pc[:, 3]
    # las.intensity = pc[:, 4]
    # las.ndvi = pc[:, 9]
    # las.hag = pc[:, 10]  # HAG
    # las.constr_flag = pc[:, 11]

    las.write(os.join(path, filename + ".las"))


def pc_normalize_neg_one(pc):
    """
    Normalize between -1 and 1
    [npoints, dim]
    """
    pc[:, 0] = pc[:, 0] * 2 - 1
    pc[:, 1] = pc[:, 1] * 2 - 1
    return pc


def rm_padding(preds, targets):
    mask = targets != -1
    targets = targets[mask]
    preds = preds[mask]

    return preds, targets, mask


def transform_2d_img_to_point_cloud(img):
    img_array = np.asarray(img)
    indices = np.argwhere(img_array > 127)
    for i in range(2):
        indices[i] = (indices[i] - img_array.shape[i] / 2) / img_array.shape[i]
    return indices.astype(np.float32)


def save_checkpoint_segmen_model(name, task, epoch, epochs_since_improvement, base_pointnet, segmen_model, opt_pointnet,
                                 opt_segmen, accuracy, batch_size, learning_rate, number_of_points):
    state = {
        'base_pointnet': base_pointnet.state_dict(),
        'segmen_net': segmen_model.state_dict(),
        'opt_pointnet': opt_pointnet.state_dict(),
        'opt_segmen': opt_segmen.state_dict(),
        'task': task,
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': number_of_points,
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'accuracy': accuracy,
    }
    filename = 'model_' + name + '.pth'
    torch.save(state, 'src/checkpoints/' + filename)


def save_checkpoint(name, epoch, epochs_since_improvement, model, optimizer, accuracy, batch_size,
                    learning_rate, n_points, weighing_method=None, weights=[], label_smoothing=None,
                    color_dropout=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': n_points,
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'accuracy': accuracy,
        'weighing_method': weighing_method,
        'weights': weights,
        'label_smoothing': label_smoothing,
        'color_dropour': color_dropout,
    }
    filename = name + '.pth'
    torch.save(state, 'src/checkpoints/' + filename)


def adjust_learning_rate(optimizer, shrink_factor=0.1):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("\nDECAYING learning rate. The new lr is %f" % (optimizer.param_groups[0]['lr'],))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def get_labels(pc):
    """
    Get labels for segmentation

    Segmentation labels:
    0 -> background (other classes we're not interested)
    1 -> tower
    2 -> cables
    3 -> low vegetation
    4 -> high vegetation
    """

    segment_labels = pc[:, 3]
    segment_labels[segment_labels == 15] = 100
    segment_labels[segment_labels == 14] = 200
    segment_labels[segment_labels == 3] = 300  # low veg
    segment_labels[segment_labels == 4] = 400  # med veg
    segment_labels[segment_labels == 5] = 400
    segment_labels[segment_labels < 100] = 0
    segment_labels = (segment_labels / 100)

    labels = segment_labels.type(torch.LongTensor)  # [2048, 5]
    return labels


def rotate_point_cloud_z(batch_data, rotation_angle=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction.
        Use input angle if given.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    if not rotation_angle:
        rotation_angle = np.random.uniform() * 2 * np.pi

    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)

    return rotated_data


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: numpy array [b, n_samples, dims]
          label: numpy array [b, n_samples]
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(labels.shape[1])
    np.random.shuffle(idx)
    return data[:, idx, :], labels[:, idx], idx


def shuffle_clusters(data, labels):
    """ Shuffle data and labels.
        Input:
            # segmentation shapes : [b, n_samples, dims, w_len]
            # targets segmen: [b, n_points, w_len]
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(labels.shape[2])
    np.random.shuffle(idx)
    return data[:, :, :, idx], labels[:, :, idx]


def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]


def rotatePoint(angle, x, y):
    a = np.radians(angle)
    cosa = np.cos(a)
    sina = np.sin(a)
    x_rot = x * cosa - y * sina
    y_rot = x * sina + y * cosa
    return x_rot, y_rot


def get_max(files_path):
    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)
        hag = data_f.HeightAboveGround
        if hag.max() > max_z:
            max_z = hag.max()


def sliding_window_coords(point_cloud, stepSize_x=10, stepSize_y=10, windowSize=[20, 20], min_points=10,
                          show_prints=False):
    """
    Slide a window across the coords of the point cloud to segment objects.

    :param point_cloud:
    :param stepSize_x:
    :param stepSize_y:
    :param windowSize:
    :param min_points:
    :param show_prints:

    :return: (dict towers, dict center_w)

    Example of return:
    For each window we get the center and the points of the tower
    dict center_w = {'0': {0: [2.9919000000227243, 3.0731000006198883]},...}
    dict towers = {'0': {0: array([[4.88606837e+05, 4.88607085e+05, 4.88606880e+05, ...,]])}...}
    """
    i_w = 0
    last_w_i = 0
    towers = {}
    center_w = {}
    point_cloud = np.array(point_cloud)
    x_min, y_min, z_min = point_cloud[0].min(), point_cloud[1].min(), point_cloud[2].min()
    x_max, y_max, z_max = point_cloud[0].max(), point_cloud[1].max(), point_cloud[2].max()

    # if window is larger than actual point cloud it means that in the point cloud there is only one tower
    if windowSize[0] > (x_max - x_min) and windowSize[1] > (y_max - y_min):
        if show_prints:
            print('Window larger than point cloud')
        if point_cloud.shape[1] >= min_points:
            towers[0] = point_cloud
            # get center of window
            center_w[0] = [point_cloud[0].mean(), point_cloud[1].mean()]
            return towers, center_w
        else:
            return None, None
    else:
        for y in range(round(y_min), round(y_max), stepSize_y):
            # check if there are points in this range of y
            bool_w_y = np.logical_and(point_cloud[1] < (y + windowSize[1]), point_cloud[1] > y)
            if not any(bool_w_y):
                continue
            if y + stepSize_y > y_max:
                continue

            for x in range(round(x_min), round(x_max), stepSize_x):
                i_w += 1
                # check points i window
                bool_w_x = np.logical_and(point_cloud[0] < (x + windowSize[0]), point_cloud[0] > x)
                if not any(bool_w_x):
                    continue
                bool_w = np.logical_and(bool_w_x, bool_w_y)
                if not any(bool_w):
                    continue
                # get coords of points in window
                window = point_cloud[:, bool_w]

                if window.shape[1] >= min_points:
                    # if not first item in dict
                    if len(towers) > 0:
                        # if consecutive windows overlap
                        if last_w_i == i_w - 1:  # or last_w_i == i_w - 2:
                            # if more points in new window -> store w, otherwise do not store
                            if window.shape[1] > towers[list(towers)[-1]].shape[1]:
                                towers[list(towers)[-1]] = window
                                center_w[list(center_w)[-1]] = [window[0].mean(), window[1].mean()]

                                last_w_i = i_w
                                if show_prints:
                                    print('Overlap window %i key %i --> %s points' % (
                                        i_w, list(towers)[-1], str(window.shape)))
                        else:
                            towers[len(towers)] = window
                            center_w[len(center_w)] = [window[0].mean(), window[1].mean()]
                            last_w_i = i_w
                            if show_prints:
                                print('window %i key %i --> %s points' % (i_w, list(towers)[-1], str(window.shape)))

                    else:
                        towers[len(towers)] = window
                        center_w[len(center_w)] = [window[0].mean(), window[1].mean()]
                        last_w_i = i_w
                        if show_prints:
                            print('window %i key %i --> %s points' % (i_w, list(towers)[-1], str(window.shape)))

        return towers, center_w


def remove_outliers(files_path, max_z=100.0):
    dir_path = os.path.dirname(files_path)
    path_norm_dir = os.path.join(dir_path, 'data_without_outliers')
    if not os.path.exists(path_norm_dir):
        os.makedirs(path_norm_dir)

    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)

        try:
            # check file is not empty
            if len(data_f.x) > 0:

                points = np.vstack((data_f.x, data_f.y, data_f.HeightAboveGround, data_f.classification,
                                    data_f.intensity,
                                    data_f.return_number,
                                    data_f.red,
                                    data_f.green,
                                    data_f.blue
                                    ))

                # Remove outliers (points above max_z)
                points = points[:, points[2] <= max_z]
                # Remove points z < 0
                points = points[:, points[2] >= 0]

                if points[2].max() > max_z:
                    print('Outliers not removed correctly!!')

                if points.shape[1] > 0:
                    f_path = os.path.join(path_norm_dir, fileName)
                    with open(f_path + '.pkl', 'wb') as f:
                        pickle.dump(points, f)
            else:
                print(f'File {fileName} is empty')
        except Exception as e:
            print(f'Error {e} in file {fileName}')


def normalize_LAS_data(files_path, max_z=100.0):
    dir_path = os.path.dirname(files_path)
    path_norm_dir = os.path.join(dir_path, 'dataset_input_model')
    if not os.path.exists(path_norm_dir):
        os.makedirs(path_norm_dir)

    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)

        try:
            # check file is not empty
            if len(data_f.x) > 0:
                # normalize axes
                data_f.x = (data_f.x - data_f.x.min()) / (data_f.x.max() - data_f.x.min())
                data_f.y = (data_f.y - data_f.y.min()) / (data_f.y.max() - data_f.y.min())
                data_f.HeightAboveGround = data_f.HeightAboveGround / max_z

                points = np.vstack((data_f.x, data_f.y, data_f.HeightAboveGround, data_f.classification))

                # Remove outliers (points above max_z)
                points = points[:, points[2] <= 1]
                # Remove points z < 0
                points = points[:, points[2] >= 0]

                if points[2].max() > 1:
                    print('Outliers not removed correctly!!')

                if points.shape[1] > 0:
                    f_path = os.path.join(path_norm_dir, fileName)
                    with open(f_path + '.pkl', 'wb') as f:
                        pickle.dump(points, f)
            else:
                print(f'File {fileName} is empty')
        except Exception as e:
            print(f'Error {e} in file {fileName}')


def normalize_pickle_data(files_path, max_z=100.0, max_intensity=5000, dir_name=''):
    dir_path = os.path.dirname(files_path)
    path_out_dir = os.path.join(dir_path, dir_name)
    if not os.path.exists(path_out_dir):
        os.makedirs(path_out_dir)

    files = glob.glob(os.path.join(files_path, '*.pkl'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        with open(file, 'rb') as f:
            pc = pickle.load(f)
        # print(pc.shape)  # [1000,4]
        # try:
        # check file is not empty
        if pc.shape[0] > 0:
            # normalize axes
            pc[:, 0] = (pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())
            pc[:, 1] = (pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())
            pc[:, 2] = pc[:, 2] / max_z

            # normalize intensity
            pc[:, 4] = pc[:, 4] / max_intensity
            pc[:, 4] = np.clip(pc[:, 4], 0, max_intensity)

            # return number
            # number of returns

            # normalize color
            pc[:, 7] = pc[:, 7] / 65536.0
            pc[:, 8] = pc[:, 8] / 65536.0
            pc[:, 9] = pc[:, 9] / 65536.0

            # Remove outliers (points above max_z)
            pc = pc[pc[:, 2] <= 1]
            # Remove points z < 0
            pc = pc[pc[:, 2] >= 0]

            if pc[:, 2].max() > 1:
                print('Outliers not removed correctly!!')

            if pc.shape[0] > 0:
                f_path = os.path.join(path_out_dir, fileName)
                with open(f_path + '.pkl', 'wb') as f:
                    pickle.dump(pc, f)
        else:
            print(f'File {fileName} is empty')
        # except Exception as e:
        #     print(f'Error {e} in file {fileName}')


def fps(pc, n_samples):
    """
    points: [N, D]  array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = pc[:, :3]
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int')  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf')  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
                (points[last_added] - points[points_left]) ** 2).sum(-1)  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, dists[points_left])  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return pc[sample_inds]


def get_ndvi(nir, red):
    a = (nir - red)
    b = (nir + red)
    c = np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    return c


######################################### augmentations #########################################


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


###################################################### samplings #######################################################


def get_sampled_sequence(pc, ids, n_points):
    """
    reshape tensor into sequence of n_points

    :param pc: Input tensor float32
    :param ids: point ids int
    :param n_points: Number of points in each sequence
    :return: Sampled sequences tensor
    """

    # Shuffle
    indices = torch.randperm(pc.shape[0])
    pc = pc[indices]
    ids = ids[indices]

    # Get needed points for sampling partition
    remain = n_points - (pc.shape[0] % n_points)

    # add points cause we need multiples of npoints 8000
    if remain != n_points:
        pc = torch.cat((pc, pc[:remain, :]), dim=0)
        ids = torch.cat((ids, ids[:remain]), dim=0)

    # Add dimension with sequence
    pc = torch.unsqueeze(pc, dim=0)
    pc = pc.view(-1, n_points, pc.shape[2])

    return pc, ids


def get_sampled_sequence_np(pc, n_points):
    """

    :param pc:
    :param n_points:
    :return:
    """
    # shuffle
    np.random.shuffle(pc)

    # get needed points for sampling partition
    remain = n_points - (pc.shape[0] % n_points)
    if remain != 0:
        # Initialize the resulting tensor
        pad_tensor = pc[:remain, :]
        pc_samp = np.concatenate((pc, pad_tensor), axis=0)
    else:
        pc_samp = pc

    # add dimension with sequence
    pc_samp = np.expand_dims(pc_samp, axis=0)
    pc_samp = np.reshape(pc_samp, (-1, n_points, pc.shape[1]))

    return pc_samp


def knn_exp_prob(ref, query_mask, n_points=8000, num_samples=8):
    """
    Efficient KNN on x,y with exponential probability computation.

    Parameters:
        ref (torch.Tensor): Reference points tensor of shape (n_ref, 2).
        query_mask (torch.Tensor): bool.
        n_points (int): Number of points to sample.

    Returns:
        torch.Tensor: Sampled indices tensor of shape (n_points,).
    """
    # query (torch.Tensor): Query points tensor of shape (n_query, 2).
    query = ref[query_mask]

    # Get indices where the mask is True
    # uncertain_ind = torch.nonzero(query_mask).view(-1)

    # if query.shape[0] < n_points/4:
        # query=query[torch.randperm(query.shape[0]), :]
        # query = query[:n_points/4, :]

    # Compute the squared Euclidean distances using torch.cdist
    distances_sq = torch.cdist(query.unsqueeze(0), ref.unsqueeze(0), p=2).squeeze(0) ** 2

    # Apply exponential function
    probs = torch.exp(-5 * distances_sq) #-5
    probs /= torch.max(probs, dim=1, keepdim=True)[0]  # Normalize probabilities

    points2sample = int(np.ceil(n_points / query.shape[0]))

    # Use torch.multinomial to sample indices based on the given probabilities
    knn_indices = torch.multinomial(probs, num_samples * points2sample, replacement=True).view(num_samples, -1)

    # Concatenate uncertain indices to the sampled indices for the first half of samples
    # if uncertain_ind.shape[0] < n_points/2:
    #     knn_indices[:3, :uncertain_ind.shape[0]] = uncertain_ind.unsqueeze(0).expand(3, -1)

    return knn_indices[:, :n_points]


def knn_indices_topk(ref, query, k):
    """
    KNN using PyTorch

    Parameters:
    - ref: Reference tensor of shape [n_ref, 2]
    - query: Query tensor of shape [n_query, 2]
    - k: Number of nearest neighbors to find

    Returns:
    - indices: Indices of the nearest neighbors (shape: [n_query, k])
    """
    # Expand dimensions for broadcasting
    ref_expanded = ref.unsqueeze(0)  # [1, n_ref, 2]
    query_expanded = query.unsqueeze(1)  # [n_query, 1, 2]

    # Compute the delta (difference) between each query and reference point
    delta = query_expanded - ref_expanded  # [n_query, n_ref, 2]

    # Calculate the squared Euclidean distances
    distances_sq = torch.pow(delta, 2).sum(dim=-1)

    # Find the top k distances and their corresponding indices
    sorted_dist_sq, indices = torch.topk(-distances_sq, k, dim=-1)  # Use topk with negated distances for sorting

    return indices


class DynamicLabelSmoothingCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, smoothing_matrix):
        super(DynamicLabelSmoothingCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.smoothing_matrix = smoothing_matrix
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        # Apply label smoothing based on the confusion matrix
        smoothed_targets = torch.zeros_like(output)
        for i in range(self.num_classes):
            smoothed_targets[:, i] = (1 - self.smoothing_matrix[i, i]) * target[:, i] + self.smoothing_matrix[i, i] / (
                    self.num_classes - 1)

        return self.ce(output, smoothed_targets)
