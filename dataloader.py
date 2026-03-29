import torch
import pandas as pd
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from config import (
    DATA, DATA_SIZE, TRAIN_RATIO, TEST_RATIO, VAL_RATIO,
    TARGET_COL, BATCH_SIZE
)


def load_qm9():
    """
    Downloads (if needed) and loads the QM9 dataset, then selects a single
    regression target and shuffles the data.

    Returns:
        qm9       : shuffled QM9 dataset with a 1-D target tensor
    """
    qm9 = QM9(root=DATA)

    # Keep only the selected regression target column
    y_target = pd.DataFrame(qm9.data.y.numpy())
    qm9.data.y = torch.Tensor(y_target[TARGET_COL])

    qm9 = qm9.shuffle()
    return qm9


def split_indices(data_size=DATA_SIZE):
    """
    Computes train / test / val index boundaries.

    Returns:
        train_end, test_end, val_end  (int, int, int)
    """
    train_end = int(data_size * TRAIN_RATIO)
    test_end  = train_end + int(data_size * TEST_RATIO)
    val_end   = test_end  + int(data_size * VAL_RATIO)
    return train_end, test_end, val_end


def normalize(qm9, train_end):
    """
    Applies z-score normalization to qm9.data.y using only training-set
    statistics (no data leakage).

    Args:
        qm9        : QM9 dataset object
        train_end  : index marking the end of the training split

    Returns:
        qm9        : dataset with normalized targets
        raw_y      : clone of the original (un-normalized) targets
        data_mean  : scalar tensor — training mean
        data_std   : scalar tensor — training std
    """
    raw_y     = qm9.data.y.clone()
    data_mean = qm9.data.y[:train_end].mean()
    data_std  = qm9.data.y[:train_end].std()
    qm9.data.y = (qm9.data.y - data_mean) / data_std
    return qm9, raw_y, data_mean, data_std


def get_dataloaders(qm9, train_end, test_end, val_end):
    """
    Wraps the three dataset splits into DataLoaders.

    Args:
        qm9        : normalized QM9 dataset
        train_end  : end index for training split
        test_end   : end index for test split
        val_end    : end index for validation split

    Returns:
        train_loader, test_loader, val_loader
    """
    train_loader = DataLoader(qm9[:train_end],        batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(qm9[train_end:test_end], batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(qm9[test_end:val_end],   batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, test_loader, val_loader


def prepare_data():
    """
    Convenience wrapper — runs the full pipeline and returns everything
    needed for training.

    Returns:
        qm9, raw_y, data_mean, data_std,
        train_loader, test_loader, val_loader
    """
    qm9                        = load_qm9()
    train_end, test_end, val_end = split_indices()
    qm9, raw_y, data_mean, data_std = normalize(qm9, train_end)
    train_loader, test_loader, val_loader = get_dataloaders(qm9, train_end, test_end, val_end)
    return qm9, raw_y, data_mean, data_std, train_loader, test_loader, val_loader