import logging
import os

import numpy as np
import torch

from src.models.pointnet2_sem_seg import PointNet2


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
        'color_dropout': color_dropout,  # checkpoints saved before 2026-11 use the misspelled key 'color_dropour'
    }
    filename = name + '.pth'
    torch.save(state, 'src/checkpoints/' + filename)


def weights_init(m):
    """Xavier init of the Conv2d layers (set-abstraction MLPs), used before training from scratch."""
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def load_model(model_checkpoint, n_classes, device):
    """Load a PointNet2 checkpoint (saved by save_checkpoint) for inference, in eval mode."""
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model = PointNet2(num_classes=n_classes).to(device)
    model.load_state_dict(checkpoint['model'])

    weights = checkpoint['weights']
    logging.info(f'Model: {os.path.basename(model_checkpoint)}, trained for {checkpoint["epoch"]} epochs')
    logging.info(f'Class weights: {None if weights is None else weights.cpu().numpy()}')
    logging.info(f'Batch size: {checkpoint["batch_size"]}, learning rate: {checkpoint["lr"]}, '
                 f'number of points: {checkpoint["number_of_points"]}')
    logging.info(f'Parameters: {sum(p.numel() for p in model.parameters())}')
    return model.eval()


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

    # pad with repeated points so the length is a multiple of n_points
    if remain != n_points:
        # wrap around so padding also works when the cloud has fewer than n_points / 2 points
        pad = torch.arange(remain) % pc.shape[0]
        pc = torch.cat((pc, pc[pad, :]), dim=0)
        ids = torch.cat((ids, ids[pad]), dim=0)

    # Add dimension with sequence
    pc = torch.unsqueeze(pc, dim=0)
    pc = pc.view(-1, n_points, pc.shape[2])

    return pc, ids


def knn_exp_prob(ref, query_mask, n_points=8000, num_samples=8):
    """
    Efficient KNN on x,y with exponential probability computation.

    Parameters:
        ref (torch.Tensor): x,y of the window points, shape (n_ref, 2).
        query_mask (torch.Tensor): bool mask of the uncertain points, shape (n_ref,).
        n_points (int): Number of points in each sample.
        num_samples (int): Number of samples to draw.

    Returns:
        torch.Tensor: Sampled indices into ref, shape (num_samples, n_points).
    """
    query = ref[query_mask]

    # Squared Euclidean distances from each uncertain point to every point of the window
    distances_sq = torch.cdist(query.unsqueeze(0), ref.unsqueeze(0), p=2).squeeze(0) ** 2

    # Sampling probability decays exponentially with distance (normalized so the closest point has prob 1)
    probs = torch.exp(-5 * distances_sq)
    probs /= torch.max(probs, dim=1, keepdim=True)[0]

    points2sample = int(np.ceil(n_points / query.shape[0]))
    knn_indices = torch.multinomial(probs, num_samples * points2sample, replacement=True).view(num_samples, -1)

    return knn_indices[:, :n_points]
