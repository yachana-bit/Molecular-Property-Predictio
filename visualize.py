import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from config import FEATURE_NAMES

def plot_target_distribution(raw_y, norm_y, data_mean, data_std):
    """
    Side-by-side histograms of the regression target before and after
    z-score normalization.

    Args:
        raw_y     : un-normalized target tensor (saved before normalization)
        norm_y    : normalized target tensor
        data_mean : training mean used for normalization
        data_std  : training std used for normalization
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Target Distribution — Dipole Moment (μ)", fontsize=13, fontweight="bold")

    axes[0].hist(raw_y.numpy(), bins=60, color="#1f77b4", edgecolor="none", alpha=0.85)
    axes[0].set_title("Before normalization")
    axes[0].set_xlabel("μ (Debye)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(data_mean.item(), color="red", linewidth=1.5, linestyle="--",
                    label=f"Mean = {data_mean:.2f}")
    axes[0].legend(fontsize=9)

    axes[1].hist(norm_y.numpy(), bins=60, color="#2ca02c", edgecolor="none", alpha=0.85)
    axes[1].set_title("After z-score normalization")
    axes[1].set_xlabel("μ (normalized)")
    axes[1].set_ylabel("Count")
    axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--", label="Mean = 0")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_node_features(qm9):
    """
    3×4 grid of histograms / bar charts for each of the 11 node features.
    Discrete features (≤10 unique values) use bar charts; continuous ones
    use histograms.

    Args:
        qm9 : loaded QM9 dataset object
    """
    x = qm9.data.x.numpy()
    n_features = x.shape[1]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle("QM9 Node Feature Distributions (all atoms)", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for i in range(n_features):
        col         = x[:, i]
        unique_vals = np.unique(col)

        if len(unique_vals) <= 10:
            counts = {v: (col == v).sum() for v in unique_vals}
            axes[i].bar(
                [str(int(k)) for k in counts.keys()],
                list(counts.values()),
                color="#1f77b4", alpha=0.85, edgecolor="none"
            )
        else:
            axes[i].hist(col, bins=40, color="#ff7f0e", edgecolor="none", alpha=0.85)

        label = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"Feature {i}"
        axes[i].set_title(label, fontsize=10)
        axes[i].set_ylabel("Count", fontsize=8)
        axes[i].tick_params(labelsize=8)

    axes[n_features].set_visible(False)   # hide unused 12th cell

    plt.tight_layout()
    plt.show()


def plot_graph_stats(qm9, sample_size=10_000):
    """
    Three histograms: atoms per molecule, edges per molecule, and average
    node degree — sampled from the dataset for speed.

    Args:
        qm9         : loaded QM9 dataset object
        sample_size : number of graphs to sample
    """
    indices = np.random.choice(len(qm9), size=min(sample_size, len(qm9)), replace=False)
    n_atoms, n_edges, avg_degrees = [], [], []

    for i in indices:
        g = qm9[i]
        num_nodes  = g.num_nodes
        num_edges  = g.edge_index.shape[1]
        n_atoms.append(num_nodes)
        n_edges.append(num_edges)
        avg_degrees.append(num_edges / num_nodes if num_nodes > 0 else 0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("QM9 Graph Structure Statistics", fontsize=13, fontweight="bold")

    for ax, data, xlabel, color in zip(
        axes,
        [n_atoms, n_edges, avg_degrees],
        ["Number of atoms", "Number of edges", "Avg degree"],
        ["#9467bd", "#8c564b", "#e377c2"],
    ):
        ax.hist(data, bins=40, color=color, edgecolor="none", alpha=0.85)
        ax.axvline(np.mean(data), color="red", linewidth=1.5, linestyle="--",
                   label=f"Mean = {np.mean(data):.1f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    axes[0].set_title("Atoms per molecule")
    axes[1].set_title("Edges per molecule")
    axes[2].set_title("Average node degree")

    plt.tight_layout()
    plt.show()

    print(f"Atoms/mol  — mean: {np.mean(n_atoms):.1f}, min: {min(n_atoms)}, max: {max(n_atoms)}")
    print(f"Edges/mol  — mean: {np.mean(n_edges):.1f}, min: {min(n_edges)}, max: {max(n_edges)}")
    print(f"Avg degree — mean: {np.mean(avg_degrees):.2f}")


def plot_split_summary(train_loader, test_loader, val_loader):
    """
    Histograms of the normalized target across the three dataset splits,
    confirming that the shuffle distributed values evenly.

    Args:
        train_loader : DataLoader for training set
        test_loader  : DataLoader for test set
        val_loader   : DataLoader for validation set
    """
    def collect(loader):
        targets = []
        for batch in loader:
            targets.extend(batch.y.numpy().tolist())
        return np.array(targets)

    splits = {
        "Train (80%)": (collect(train_loader), "#1f77b4"),
        "Test  (10%)": (collect(test_loader),  "#d62728"),
        "Val   (10%)": (collect(val_loader),   "#2ca02c"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    fig.suptitle("Normalized Target Distribution per Split", fontsize=13, fontweight="bold")

    for ax, (label, (y, color)) in zip(axes, splits.items()):
        ax.hist(y, bins=50, color=color, edgecolor="none", alpha=0.85)
        ax.set_title(f"{label}\nn={len(y):,}  mean={y.mean():.3f}  std={y.std():.3f}")
        ax.set_xlabel("μ (normalized)")
        ax.set_ylabel("Count")
        ax.axvline(y.mean(), color="black", linewidth=1.2, linestyle="--")

    plt.tight_layout()
    plt.show()


def plot_feature_correlation(qm9, sample_atoms=20_000):
    """
    Pearson correlation heatmap across all 11 node features.

    Args:
        qm9          : loaded QM9 dataset object
        sample_atoms : number of atom rows to sample (full set is large)
    """
    x = qm9.data.x.numpy()
    if x.shape[0] > sample_atoms:
        idx = np.random.choice(x.shape[0], sample_atoms, replace=False)
        x   = x[idx]

    df   = pd.DataFrame(x, columns=FEATURE_NAMES)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle("Node Feature Correlation Matrix", fontsize=13, fontweight="bold")

    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=9)

    for i in range(len(FEATURE_NAMES)):
        for j in range(len(FEATURE_NAMES)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "black")

    plt.tight_layout()
    plt.show()


def visualize_dataset(qm9, raw_y, train_loader, test_loader, val_loader, data_mean, data_std):
    """
    Master function — runs all 5 dataset visualizations in sequence.

    Args:
        qm9          : QM9 dataset object (after normalization)
        raw_y        : raw (un-normalized) target tensor
        train_loader : DataLoader
        test_loader  : DataLoader
        val_loader   : DataLoader
        data_mean    : normalization mean
        data_std     : normalization std
    """
    print("1/5  Target distribution...")
    plot_target_distribution(raw_y, qm9.data.y, data_mean, data_std)

    print("2/5  Node feature distributions...")
    plot_node_features(qm9)

    print("3/5  Graph structure stats...")
    plot_graph_stats(qm9)

    print("4/5  Split comparison...")
    plot_split_summary(train_loader, test_loader, val_loader)

    print("5/5  Feature correlations...")
    plot_feature_correlation(qm9)

    print("Done.")


#  TRAINING DIAGNOSTICS
def plot_loss(gcn_train_loss, gcn_val_loss, gin_train_loss, gin_val_loss):
    """
    Overlaid train/val loss curves for GCN and GIN.

    Args:
        gcn_train_loss : ndarray of GCN training losses per epoch
        gcn_val_loss   : ndarray of GCN validation losses per epoch
        gin_train_loss : ndarray of GIN training losses per epoch
        gin_val_loss   : ndarray of GIN validation losses per epoch
    """
    plt.figure(figsize=(8, 5))
    plt.plot(gcn_train_loss, label="Train loss (GCN)")
    plt.plot(gcn_val_loss,   label="Val loss (GCN)")
    plt.plot(gin_train_loss, label="Train loss (GIN)")
    plt.plot(gin_val_loss,   label="Val loss (GIN)")
    plt.legend()
    plt.ylabel("Loss (MSE)")
    plt.xlabel("Epoch")
    plt.title("Model Loss over Epochs")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()


def plot_targets(pred, ground_truth, title="Ground truth vs Prediction"):
    """
    Scatter plot of predicted values vs ground truth, with a perfect-fit
    diagonal line.

    Args:
        pred         : ndarray of model predictions
        ground_truth : ndarray of true target values
        title        : plot title
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred, ground_truth, s=0.5, alpha=0.6)
    ax.axline((1, 1), slope=1, color="red", linewidth=1.2, linestyle="--", label="Perfect fit")
    plt.xlim(-2, 7)
    plt.ylim(-2, 7)
    plt.xlabel("Predicted value")
    plt.ylabel("Ground truth")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()