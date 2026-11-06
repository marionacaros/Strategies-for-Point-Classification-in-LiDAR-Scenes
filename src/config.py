"""Settings shared by the training and test scripts."""

# Segmentation labels (see src/datasets.py):
#   0 ground (not evaluated), 1 transmission tower, 2 power lines, 3 surrounding, 4 wind turbine
N_CLASSES = 5

# Probability of zeroing the colour/NDVI channels (4:) of a training batch
COLOR_DROPOUT = 0.2

# Class weighting for the CE loss: 'EFS', 'INS', 'ISNS', 'sklearn' (computed from SAMPLES_X_CLASS_CAT3
# with utils.get_metrics.get_weights4class) or None. Explicit weights can be passed with
# --class_weights in src/train_segmentation_pn2.py instead.
WEIGHING_METHOD = None

# Points per class in the training set, in label order. Needed only when WEIGHING_METHOD is set.
# TODO: fill in from the training data (see TASKS.md).
SAMPLES_X_CLASS_CAT3 = None
