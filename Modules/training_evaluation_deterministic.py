
#################################################################################################################
                                        # Import libraries and modules
#################################################################################################################


import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import os
import sys
from torch import nn

# Set current and repository working directory
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from Modules.Early_Stopping import *




#################################################################################################################
                            # Training Functions for Sequential MLP and MLP
#################################################################################################################


def train_model_deterministic(model, train_loader, val_loader,
                              optimizer, num_epochs=1000, device='cpu',
                              early_stopping=True, patience=15, delta=1e-4,
                              print_every=25, verbose=True, eval_every=2):
    """
    Train a deterministic model with optional early stopping

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training set.
        val_loader (DataLoader): Validation set.
        optimizer (Optimizer): Optimizer.
        num_epochs (int): Max number of epochs.
        device (str): 'cuda' or 'cpu'.
        early_stopping (bool): Whether to use early stopping.
        patience (int): Patience for early stopping.
        delta (float): Min improvement for early stopping.
        print_every (int): Print loss every N epochs.
        verbose (bool): Whether to print logs.
        eval_every (int): Evaluate on validation set every N epochs.

    Returns:
        dict: Training-Validation loss history
        - 'epoch': List of epochs.
        - 'train_loss': List of training losses.
        - 'val_loss': List of validation losses recorded at the epochs in 'val_epoch' (present only if early stopping is used).
        - 'val_epoch': List of epoch indices where validation was computed (present only if early stopping is used).
        - 'best_epoch': Best epoch where validation improved the most (None if early stopping is False or no validation ran).
        - 'best_val': Best validation loss (None if early stopping is False or no validation ran).
        - 'best_train_at_best': Training loss at best_epoch (None if early stopping is False or no validation ran).

    """
    model.to(device)
    model.train()
    val_loss = float("nan")

    # Initialize early stopping
    stopper = EarlyStopping(patience=patience, delta=delta) if early_stopping else None

    # Define criterion
    criterion= nn.MSELoss()

    # Track best epoch info
    best_epoch = None
    best_val = None
    best_train_at_best = None

    # Initialize dictionary to store training validation loss history
    train_val_history = {"epoch": [], "train_loss": [], "val_loss": [], "val_epoch": []}


    # Remind users about scale of losses
    if verbose:
        print("Note: Training and validation losses are reported on the standardized scale of the target variable.")

    # Training loop over epochs
    for epoch in range(num_epochs):
        total_loss = 0.0
        model.train()
        
        # Iterate over batches
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            _, y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        train_val_history["epoch"].append(epoch + 1)
        train_val_history["train_loss"].append(avg_loss)
        
        # Early stopping validation
        if stopper is not None and (epoch + 1) % eval_every == 0:
            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for X_val, y_val in val_loader:
                    X_val, y_val = X_val.to(device), y_val.to(device)
                    _, y_val_pred = model(X_val)
                    val_loss += criterion(y_val_pred, y_val).item()
            val_loss /= len(val_loader)
            train_val_history["val_loss"].append(val_loss)
            train_val_history["val_epoch"].append(epoch + 1)

            # Compute early stopping condition and break loop if met
            prev_best = stopper.best_loss
            stopper(val_loss, model)

            # Record best epoch info immediately when an improvement happens
            if stopper.best_loss is not None and (prev_best is None or stopper.best_loss < prev_best - 1e-12):
                best_epoch = epoch + 1
                best_val = stopper.best_loss
                best_train_at_best = avg_loss

            if stopper.early_stop:
                break
        
        if verbose and ((epoch + 1) % print_every == 0 or epoch == 0):
            if early_stopping:
                val_loss_str = f"{val_loss:.5f}" if not np.isnan(val_loss) else "N/A"
                best_loss_str = f"{stopper.best_loss:.5f}" if stopper.best_loss is not None else "N/A"
                print(f"Epoch [{epoch+1}/{num_epochs}] — Train Loss: {avg_loss:.5f} | Val Loss: {val_loss_str} | Best Val Loss: {best_loss_str}")
            else:
                print(f"Epoch [{epoch+1}/{num_epochs}] — Train Loss: {avg_loss:.5f}")

    # Load best model among epochs
    if stopper is not None:
        stopper.load_best_model(model)

    # Final summary
    if verbose:
        if early_stopping and best_epoch is not None:
            print(f"\nBest Epoch {best_epoch} - MSE Train Loss: {best_train_at_best:.5f}")
            print(f"MSE Validation Loss: {best_val:.5f}")
        else:
            print(f"\nFinal Epoch {num_epochs} - MSE Train Loss: {train_val_history['train_loss'][-1]:.5f}")

    # Remove validation loss if early stopping is not used
    if not early_stopping:
        train_val_history.pop("val_loss")
        train_val_history.pop("val_epoch", None)

    # Attach best info for downstream use in the dictionary
    train_val_history["best_epoch"] = best_epoch
    train_val_history["best_val"] = best_val
    train_val_history["best_train_at_best"] = best_train_at_best

    return train_val_history


#################################################################################################################
                                        # Evaluation Functions for SML and MLP
#################################################################################################################


def evaluate_model_deterministic(
    model,
    train_loader,
    test_loader,
    scaler_y=None,
    return_preds=False,
    upperquantile_mse=None,
    device="cpu",
    verbose=False,
    return_r2=False
):
    
    """ Evaluate a deterministic model on a held out test set and report metrics on the original target scale. This procedure feeds the test inputs through the model once, maps predictions and targets back to the original scale if a scaler is supplied, and computes standard pointwise accuracy measures. Tail performance is characterized by a single upper quantile mean squared error computed above a threshold estimated from the training target distribution.

    Args: 
    model (nn.Module): Trained deterministic model whose forward method returns a tuple (hidden_out, y_pred). 
    train_loader (DataLoader): Training data loader used only to estimate the extreme target threshold on the original scale for the upper quantile error. 
    test_loader (DataLoader): Test data loader over which evaluation metrics are computed. 
    scaler_y (StandardScaler or None): Optional target scaler fitted during training. If provided, predictions and targets are inverse transformed to the original scale before computing metrics. If None, all values are assumed to already be on the original scale. 
    return_preds (bool): If True, return arrays of predictions and ground truth on the original scale for downstream analysis and plotting. 
    upperquantile_mse (float or None): If a float strictly between zero and one is provided, compute the mean squared error of the prediction restricted to test observations whose true value lies above the corresponding quantile of the training target distribution on the original scale. If None, this tail metric is skipped. 
    device (str): Device used for evaluation, typically "cpu" or "cuda". 
    verbose (bool): If True, print a concise textual summary of the computed metrics. 
    return_r2 (bool): If True, compute and return the coefficient of determination R squared on the original scale. When the variance of the targets is zero, R squared is set to NaN.

    Returns: 
    dict: Dictionary with the following fields on the original scale with the following key. 
        "mse": Mean squared error computed over the full test set. 
        "mae_mean": Mean absolute error computed over the full test set. 
        "upperquantile_mse": Mean squared error on test observations above the training based threshold, or NaN when the set is empty or the option is disabled. 
        "upperquantile_threshold": Numerical value of the threshold on the original scale used to define the extreme set, or NaN when not computed. 
        "upperquantile_count": Number of test observations strictly above the threshold, equal to zero when not computed. 
        "r2": Present only when return_r2 is True. Coefficient of determination on the original scale, or NaN when undefined. 
        "y_pred" and "y_true": Present only when return_preds is True. Arrays of predictions and ground truth on the original scale for downstream analysis. 
        
    """


    # Move model to device and stop tracking gradients
    model.to(device)
    model.eval()

    # Collect predictions and targets on the test set
    all_y_true = []
    all_y_pred = []

    # Compute full prediction set
    with torch.inference_mode():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            _, y_pred = model(x_batch)

            # Append prediction and true values
            all_y_true.append(y_batch.detach().cpu().numpy().flatten())
            all_y_pred.append(y_pred.detach().cpu().numpy().flatten())

    # Concatenate across batches
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    # Inverse-transform to original scale if a scaler is provided
    if scaler_y is not None:
        y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_orig = y_true
        y_pred_orig = y_pred

    # Base deterministic metrics on the original scale
    mse = mean_squared_error(y_true_orig, y_pred_orig)
    mae_mean = mean_absolute_error(y_true_orig, y_pred_orig)

    result = {
        "mse": mse,
        "mae_mean": mae_mean
    }

    # Compute R^2 metrics
    if return_r2:
        r2 = r2_score(y_true_orig, y_pred_orig)
        result["r2"] = r2

    # Initialize upper-quantile fields
    extreme_count = 0
    q_thresh = np.nan
    result["upperquantile_mse"] = np.nan
    result["upperquantile_threshold"] = np.nan
    result["upperquantile_count"] = 0

    # Compute mse restricted to the upper quantile of the training target distribution if requested
    if isinstance(upperquantile_mse, float) and 0 < upperquantile_mse < 1:
        
        # Extract responses from train dataloader
        y_train_tensor = train_loader.dataset.tensors[1]
        if y_train_tensor.is_cuda:
            y_train = y_train_tensor.detach().cpu().numpy().reshape(-1)
        else:
            y_train = y_train_tensor.detach().numpy().reshape(-1)

        # Map training targets to the original scale if a scaler was used
        if scaler_y is not None:
            y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
        else:
            y_train_orig = y_train

        # Estimate the quantile threshold on the original scale from training targets
        q_thresh = np.quantile(y_train_orig, upperquantile_mse)

        # Identify test observations strictly above the threshold on the original scale
        extreme_mask = y_true_orig > q_thresh
        extreme_count = int(np.sum(extreme_mask))

        if extreme_count > 0:
            mse_extreme = mean_squared_error(y_true_orig[extreme_mask], y_pred_orig[extreme_mask])
        else:
            mse_extreme = np.nan

        # Store results
        result["upperquantile_mse"] = mse_extreme
        result["upperquantile_threshold"] = float(q_thresh)
        result["upperquantile_count"] = extreme_count
        
        # Store empty result if not needed
    else:
        result["upperquantile_mse"] = np.nan
        result["upperquantile_threshold"] = np.nan
        result["upperquantile_count"] = 0


    # Optionally store all predictions and true values on the original scale
    if return_preds:
        result["y_pred"] = y_pred_orig
        result["y_true"] = y_true_orig

    # Print summary evaluation
    if verbose:
        print("Evaluation metrics on the original scale")
        print(f"RMSE: {np.sqrt(mse):.5f}")
        print(f"MAE (mean): {mae_mean:.5f}")
        if "r2" in result:
            if np.isnan(result["r2"]):
                print("R^2: NaN")
            else:
                print(f"R^2: {result['r2']:.5f}")
        if isinstance(upperquantile_mse, float) and 0 < upperquantile_mse < 1:
            if extreme_count > 0 and not np.isnan(result["upperquantile_mse"]):
                print(
                    f"RMSE on test samples above training q={upperquantile_mse:.3f} "
                    f"threshold {q_thresh:.5f} "
                    f"({extreme_count} observations): {np.sqrt(result['upperquantile_mse']):.5f}"
                )
            else:
                print(
                    f"No test observations above training q={upperquantile_mse:.3f} "
                    f"threshold {q_thresh:.5f}. upperquantile_mse set to NaN."
                )

    return result


#################################################################################################################