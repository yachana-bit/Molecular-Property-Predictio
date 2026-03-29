import torch

from config import (
    EPOCHS,
    GCN_DIM_H, GIN_DIM_H,
    GCN_MODEL_PATH, GIN_MODEL_PATH,
)
from data_loader import prepare_data
from models import GCN, GIN
from train import train_epochs, test
from visualize import (
    visualize_dataset,
    plot_loss,
    plot_targets,
)


def main():
    # Load and prepare data
    print("Loading QM9 dataset...")
    qm9, raw_y, data_mean, data_std, train_loader, test_loader, val_loader = prepare_data()
    print(f"Dataset ready — train: {len(train_loader.dataset):,}  "
          f"test: {len(test_loader.dataset):,}  val: {len(val_loader.dataset):,}")

    # Dataset visualization
    print("\nVisualizing dataset...")
    visualize_dataset(qm9, raw_y, train_loader, test_loader, val_loader, data_mean, data_std)

    # Train GCN
    print("\nTraining GCN...")
    gcn_model = GCN(dim_h=GCN_DIM_H)
    gcn_train_loss, gcn_val_loss, gcn_train_pred, gcn_train_true = train_epochs(
        EPOCHS, gcn_model, train_loader, val_loader, GCN_MODEL_PATH
    )

    # Train GIN 
    print("\nTraining GIN...")
    gin_model = GIN(dim_h=GIN_DIM_H)
    gin_train_loss, gin_val_loss, gin_train_pred, gin_train_true = train_epochs(
        EPOCHS, gin_model, train_loader, val_loader, GIN_MODEL_PATH
    )

    # Training diagnostics
    print("\nPlotting training curves...")
    plot_loss(gcn_train_loss, gcn_val_loss, gin_train_loss, gin_val_loss)

    plot_targets(gin_train_pred, gin_train_true, title="GIN — train: ground truth vs prediction")

    #  Evaluate GCN on test set
    print("\nEvaluating GCN on test set...")
    gcn_model.load_state_dict(torch.load(GCN_MODEL_PATH, weights_only=True))
    gcn_test_loss, gcn_test_pred, gcn_test_true = test(test_loader, gcn_model)
    print(f"GCN test loss: {gcn_test_loss.item():.4f}")
    plot_targets(gcn_test_pred, gcn_test_true, title="GCN — test: ground truth vs prediction")

    # Evaluate GIN on test set 
    print("\nEvaluating GIN on test set...")
    gin_model.load_state_dict(torch.load(GIN_MODEL_PATH, weights_only=True))
    gin_test_loss, gin_test_pred, gin_test_true = test(test_loader, gin_model)
    print(f"GIN test loss: {gin_test_loss.item():.4f}")
    plot_targets(gin_test_pred, gin_test_true, title="GIN — test: ground truth vs prediction")


if __name__ == "__main__":
    main()