import os

import numpy as np
import torch

# Classes that are evaluated. Ground (label 0) is not a target class: it is neither scored nor part of the mIoU.
EVAL_CLASSES = {1: 'tower', 2: 'lines', 3: 'surrounding', 4: 'wind_turbine'}
RESULTS_HEADER = 'dataset,n_points,' + ','.join(f'IoU_{name}' for name in EVAL_CLASSES.values()) + ',mIoU,time_min\n'


def get_iou_obj(pc_preds: torch.LongTensor, targets: torch.LongTensor, label: int = 1):
    # get IoU
    corrects = pc_preds == targets
    gt_positive = torch.sum(targets == label).item()
    detected_positive = (pc_preds == label).sum().item()
    tp = (corrects & (pc_preds == label)).sum().item()
    fp = detected_positive - tp

    # Handle division by zero
    if gt_positive + fp == 0:
        iou_obj = torch.nan
    else:
        iou_obj = tp / (gt_positive + fp)
    return iou_obj


def get_iou_np(pc_preds, targets, label = 1):
    """
    IoU of one class (numpy).

    NOTE: an IoU of exactly 0 is returned as NaN (kept on purpose to reproduce the paper's numbers),
    so a class that is completely missed is skipped by np.nanmean instead of lowering the mIoU.
    NaN is also returned when the class is absent from both targets and predictions.
    """
    corrects = pc_preds == targets
    gt_positive = np.sum(targets == label)
    detected_positive = np.count_nonzero(pc_preds == label)
    tp = np.count_nonzero(corrects & (pc_preds == label))
    fp = detected_positive - tp

    # Handle division by zero
    if gt_positive + fp == 0:
        iou_obj = np.nan
    else:
        iou_obj = tp / (gt_positive + fp)

    if iou_obj == 0.:
        iou_obj = np.nan

    return iou_obj


def get_ious(targets, preds):
    """IoU of each evaluated class plus their mean ('mIoU'), as fractions (numpy arrays of labels)."""
    iou = {name: get_iou_np(preds, targets, label) for label, name in EVAL_CLASSES.items()}
    iou['mIoU'] = np.nanmean(list(iou.values()))
    return iou


def print_ious(iou):
    for name, value in iou.items():
        print(f'IoU {name}: {value * 100:.2f}')


def append_results_csv(path, dataset, n_points, iou, time_min):
    """Append one row (IoUs in %) to the results CSV, writing the header if the file is new."""
    write_header = not os.path.exists(path)
    values = [f'{iou[name] * 100:.2f}' for name in list(EVAL_CLASSES.values()) + ['mIoU']]
    with open(path, 'a') as fid:
        if write_header:
            fid.write(RESULTS_HEADER)
        fid.write(','.join([dataset, str(n_points)] + values + [str(time_min)]) + '\n')


def get_accuracy(preds, targets, metrics):
    corrects = torch.eq(preds.view(-1), targets.view(-1))
    if len(corrects) != 0:
        metrics['accuracy'] = (corrects.sum().item() / len(corrects))
    else:
        metrics['accuracy'] = 0
    return metrics


def get_weights_effective_num_of_samples(n_of_classes, beta, samples_per_cls):
    """The authors suggest experimenting with different beta values: 0.9, 0.99, 0.999, 0.9999."""
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights4class = (1.0 - beta) / np.array(effective_num)
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_inverse_num_of_samples(n_of_classes, samples_per_cls, power=1.0):
    weights4class = 1.0 / np.array(np.power(samples_per_cls, power))  # [0.03724195 0.00244003]
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_sklearn(n_of_classes, samples_per_cls):
    weights4class = np.sum(samples_per_cls) / np.multiply(n_of_classes, samples_per_cls)
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights4class(weighing_method, n_classes, samples_per_cls, beta=None):
    """

       :param weighing_method: str, options available: "EFS" "INS" "ISNS"
       :param n_classes: int, representing the total number of classes in the entire train set
       :param samples_per_cls: A python list of size [n_classes]
       :param beta: float,

       :return weights4class: torch.Tensor of size [batch, n_classes]
    """
    if weighing_method == 'EFS':
        weights4class = get_weights_effective_num_of_samples(n_classes, beta, samples_per_cls)
    elif weighing_method == 'INS':
        weights4class = get_weights_inverse_num_of_samples(n_classes, samples_per_cls)
    elif weighing_method == 'ISNS':
        weights4class = get_weights_inverse_num_of_samples(n_classes, samples_per_cls, 0.5)  # [0.9385, 0.0615]
    elif weighing_method == 'sklearn':
        weights4class = get_weights_sklearn(n_classes, samples_per_cls)
    else:
        return None

    weights4class = torch.tensor(weights4class).float()
    return weights4class
