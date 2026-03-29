import math
import numpy as np
import torch
import torch.nn as nn

from config import LEARNING_RATE, WEIGHT_DECAY


def train_one_epoch(loader, model, loss_fn, optimizer):
    """
    Runs one full training epoch over the DataLoader.

    Args:
        loader    : DataLoader for training data
        model     : GNN model
        loss_fn   : loss function (e.g. MSELoss)
        optimizer : torch optimizer

    Returns:
        epoch_loss (Tensor) : mean loss over all batches
        model               : updated model
    """
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()
        batch.x = batch.x.float()

        out  = model(batch)
        loss = loss_fn(out, batch.y.reshape(-1, 1))

        total_loss += loss / len(loader)
        loss.backward()
        optimizer.step()

    return total_loss, model


def validate(loader, model, loss_fn):
    """
    Evaluates the model on a validation DataLoader without updating weights.

    Args:
        loader  : DataLoader for validation data
        model   : current model
        loss_fn : loss function

    Returns:
        val_loss (Tensor)
    """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in loader:
            out      = model(batch)
            val_loss += loss_fn(out, batch.y.reshape(-1, 1)) / len(loader)

    return val_loss


@torch.no_grad()
def test(loader, model):
    """
    Runs inference on the test set and collects predictions vs ground truth.

    Args:
        loader : DataLoader for test data
        model  : trained model

    Returns:
        test_loss      (float)   : mean MSE over test set
        predictions    (ndarray) : model output values
        ground_truth   (ndarray) : true target values
    """
    loss_fn   = nn.MSELoss()
    test_loss = 0
    predictions  = np.empty(0)
    ground_truth = np.empty(0)

    for batch in loader:
        out       = model(batch)
        test_loss += loss_fn(out, batch.y.reshape(-1, 1)) / len(loader)
        predictions  = np.concatenate((predictions,  out.numpy()[:, 0]))
        ground_truth = np.concatenate((ground_truth, batch.y.numpy()))

    return test_loss, predictions, ground_truth


def train_epochs(epochs, model, train_loader, val_loader, save_path):
    """
    Full training loop over multiple epochs with best-model checkpointing.

    Args:
        epochs       : number of epochs
        model        : GNN model to train
        train_loader : DataLoader for training data
        val_loader   : DataLoader for validation data
        save_path    : file path to save the best model checkpoint

    Returns:
        train_losses      (ndarray) : train loss per epoch
        val_losses        (ndarray) : val loss per epoch
        train_predictions (ndarray) : predictions on train set (last epoch)
        train_targets     (ndarray) : ground truth on train set (last epoch)
    """
    optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn    = nn.MSELoss()

    train_losses = np.empty(epochs)
    val_losses   = np.empty(epochs)
    best_loss    = math.inf
    train_predictions = np.empty(0)
    train_targets     = np.empty(0)

    for epoch in range(epochs):
        epoch_loss, model = train_one_epoch(train_loader, model, loss_fn, optimizer)
        v_loss            = validate(val_loader, model, loss_fn)

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), save_path)

        # Collect train predictions on the last epoch
        if epoch == epochs - 1:
            with torch.no_grad():
                for batch in train_loader:
                    out = model(batch)
                    train_predictions = np.concatenate((train_predictions, out.numpy()[:, 0]))
                    train_targets     = np.concatenate((train_targets,     batch.y.numpy()))

        train_losses[epoch] = epoch_loss.detach().numpy()
        val_losses[epoch]   = v_loss.detach().numpy()

        if epoch % 2 == 0:
            print(f"Epoch {epoch:>3} | Train loss: {epoch_loss.item():.4f} | Val loss: {v_loss.item():.4f}")

    return train_losses, val_losses, train_predictions, train_targets