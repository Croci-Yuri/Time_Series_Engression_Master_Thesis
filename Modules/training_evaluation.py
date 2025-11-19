#################################################################################################################
                                    # Import libraries
#################################################################################################################


import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import os
import sys

# Set current and repository working directory
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from Modules.Early_Stopping import *
from Modules.loss import multi_energy_loss


#################################################################################################################
                                    # Training function
#################################################################################################################


def train_model(model, train_loader, val_loader,
                optimizer, num_epochs, m=2,
                early_stopping=True, patience=20, delta=3e-4,
                m_validation=50, print_every=25,
                verbose=False, eval_every=2,
                device="cpu"
):
    """
    Train a stochastic Engression model with optional early stopping

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training set.
        val_loader (DataLoader): Validation set.
        optimizer (Optimizer): Optimizer.
        num_epochs (int): Max number of epochs.
        m (int): Number of samples per input for training loss.
        device (str): 'cuda' or 'cpu'.
        early_stopping (bool): Whether to use early stopping.
        patience (int): Patience for early stopping.
        delta (float): Min improvement for early stopping.
        m_validation (int): Number of samples per input for validation loss.
        print_every (int): Print loss every N epochs.
        verbose (bool): Whether to print logs.
        eval_every (int): Evaluate on validation set every N epochs.

    Returns:
        dict: Training-Validation loss history
        - 'epoch': List of epochs.
        - 'train_loss': List of training losses.
        - 's1': List of sharpness terms E(|Y - Ŷ|).
        - 's2': List of diversity terms E(|Ŷ - Ŷ′|).
        - 'val_loss': List of validation losses recorded at the epochs in 'val_epoch' (present only if early stopping is used).
        - 'val_epoch': List of epoch indices where validation was computed (present only if early stopping is used).
        - 'best_epoch': Best epoch where validation improved the most (None if early stopping is False or no validation ran).
        - 'best_val': Best validation loss (None if early stopping is False or no validation ran).
        - 'best_train_at_best': Training loss at best_epoch (None if early stopping is False or no validation ran).

    """
    # M-Samples training and validation (checker)
    if not isinstance(m, int) or m < 2:
        raise ValueError("Number of samples m must be an integer greater or equal to 2.")
    if not isinstance(m_validation, int) or m_validation < 2:
        raise ValueError("Number of samples m_validation must be an integer greater or equal to 2.")

    model.to(device)
    model.train()

    # Early stopping helper uses only patience and delta
    stopper = EarlyStopping(patience=patience, delta=delta) if early_stopping else None

    # Define criterion
    criterion = multi_energy_loss

    # Remind users about scale of losses
    if verbose:
        print("Note: Training and validation losses are reported on the standardized scale of the target variable.")

    # Initialize dictionary to store training-validation loss history
    train_val_history = {"epoch": [], "train_loss": [], "s1": [], "s2": [], "val_loss": [], "val_epoch": []}
    val_running = float("nan")

    best_epoch = None
    best_val = None
    best_train_at_best = None

    for epoch in range(num_epochs):
        model.train()
        sum_loss = 0.0
        sum_s1 = 0.0
        sum_s2 = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)

            # M stochastic draws via input expansion
            B = x_batch.size(0)
            x_expanded = x_batch.repeat_interleave(m, dim=0)
            y_expanded = model(x_expanded)              # shape (B*m, d)
            y_samples = y_expanded.view(B, m, -1)       # shape (B, m, d)

            s1, s2, loss = criterion(y_batch, y_samples)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            sum_s1 += s1.item()
            sum_s2 += s2.item()

        avg_loss = sum_loss / len(train_loader)
        avg_s1 = sum_s1 / len(train_loader)
        avg_s2 = sum_s2 / len(train_loader)

        train_val_history["epoch"].append(epoch + 1)
        train_val_history["train_loss"].append(avg_loss)
        train_val_history["s1"].append(avg_s1)
        train_val_history["s2"].append(avg_s2)

        # Validation for early stopping
        if stopper is not None and (epoch + 1) % eval_every == 0:
            model.eval()
            val_acc = 0.0
            with torch.inference_mode():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    Bv = x_val.size(0)
                    x_val_exp = x_val.repeat_interleave(m_validation, dim=0)
                    y_val_exp = model(x_val_exp).view(Bv, m_validation, -1)

                    _, _, val_energy = criterion(y_val, y_val_exp)
                    val_acc += val_energy.item()

            val_running = val_acc / len(val_loader)
            train_val_history["val_loss"].append(val_running)
            train_val_history["val_epoch"].append(epoch + 1)

            prev_best = stopper.best_loss
            stopper(val_running, model)

            if stopper.best_loss is not None and (prev_best is None or stopper.best_loss < prev_best - 1e-12):
                best_epoch = epoch + 1
                best_val = stopper.best_loss
                best_train_at_best = avg_loss

            if stopper.early_stop:
                break

        if verbose and ((epoch + 1) % print_every == 0 or epoch == 0):
            v_str = f"{val_running:.5f}" if not np.isnan(val_running) else "N/A"
            b_str = f"{stopper.best_loss:.5f}" if (stopper is not None and stopper.best_loss is not None) else "N/A"
            if early_stopping:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] — Energy Loss: {avg_loss:.5f} | "
                    f"S1: {avg_s1:.5f}, S2: {avg_s2:.5f} | "
                    f"Val Energy: {v_str} | Best Val Energy: {b_str}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] — Energy Loss: {avg_loss:.5f} | "
                    f"S1: {avg_s1:.5f}, S2: {avg_s2:.5f}"
                )

    # Reload best model when early stopping was active
    if stopper is not None:
        stopper.load_best_model(model) 

    # Final summary
    if verbose:
        if early_stopping and best_epoch is not None:
            print(
                f"\nBest Epoch {best_epoch}  Train Energy Loss: {best_train_at_best:.5f}\n"
                f"Validation Energy Loss: {best_val:.5f}"
            )
        else:
            print(
                f"\nFinal Epoch {num_epochs}  Train Energy Loss: {train_val_history['train_loss'][-1]:.5f}"
            )

    train_val_history["best_epoch"] = best_epoch
    train_val_history["best_val"] = best_val
    train_val_history["best_train_at_best"] = best_train_at_best

    # Remove unused 
    if not early_stopping:
        train_val_history.pop("val_loss", None)
        train_val_history.pop("val_epoch", None)
        

    return train_val_history



#################################################################################################################
                                    # Evaluation function
#################################################################################################################


def evaluate_model(
    model,
    train_loader,
    test_loader,
    scaler_y=None,
    m=100,
    return_preds=False,
    upperquantile_mse=None,
    coverage=0.95,
    lower_quantile=None,
    upper_quantile=None,
    device="cpu",
    verbose=False):

    """
Evaluate a stochastic Engression model on a held out test set and report metrics on the original target scale.

This procedure draws m stochastic samples per input, maps predictions and targets back to the original scale if a scaler is supplied, and computes energy loss together with standard pointwise accuracy measures. Interval quality is summarized through coverage and sharpness at a central level determined by a single coverage argument. Optional quantile prediction curves can be returned for plotting. Tail performance is characterized by a single upper quantile mean squared error computed above a threshold estimated from the training target distribution.

Args:
    model (nn.Module): Trained Engression model.
    train_loader (DataLoader): Training data loader used only to estimate the extreme target threshold on the original scale for the upper quantile error.
    test_loader (DataLoader): Test data loader over which evaluation metrics are computed.
    scaler_y (StandardScaler or None): Optional target scaler fitted during training. If provided, predictions and targets are inverse transformed to the original scale before computing metrics. If None, all values are assumed to already be on the original scale.
    m (int): Number of stochastic samples generated per input during evaluation. The value must be an integer greater or equal to two.
    return_preds (bool): If True, return arrays of predictions and ground truth on the original scale for downstream analysis and plotting.
    upperquantile_mse (float or None): If a float strictly between zero and one is provided, compute the mean squared error of the mean prediction restricted to test observations whose true value lies above the corresponding quantile of the training target distribution on the original scale. If None, this tail metric is skipped.
    coverage (float): Central coverage level strictly between zero and one used to form a symmetric interval around the median of the predictive distribution. Coverage and sharpness are always computed at this level.
    lower_quantile (float or None): Optional lower quantile in the open interval zero to one for which the corresponding prediction curve is returned when return_preds is True. This argument is not used for metric computation.
    upper_quantile (float or None): Optional upper quantile in the open interval zero to one for which the corresponding prediction curve is returned when return_preds is True. This argument is not used for metric computation.
    device (str): Device used for evaluation, typically "cpu" or "cuda".
    verbose (bool): If True, print a concise textual summary of the computed metrics.

Returns:

    dict: Dictionary with the following fields on the original scale with the following key:

        "energy_loss": Scalar energy loss defined as s1 minus one half times s2.
        "s1": Scalar value of E(|Y - Ŷ|).
        "s2": Scalar value of E(|Ŷ - Ŷ′|).
        "mse": Mean squared error of the mean prediction.
        "mae_mean": Mean absolute error of the mean prediction.
        "coverage": Empirical coverage of the central interval at the requested coverage level.
        "sharpness": Average width of that central interval.
        "upperquantile_mse": Mean squared error of the mean prediction on test observations above the training based threshold, or NaN when the set is empty or the option is disabled.
        "upperquantile_threshold": Numerical value of the threshold on the original scale used to define the extreme set, or NaN when not computed.
        "upperquantile_count": Number of test observations strictly above the threshold, equal to zero when not computed.
        The following fields are present only when return_preds is True.
        "y_pred_mean": Array of mean predictions with shape (N, d).
        "y_pred_median": Array of median predictions with shape (N, d).
        "y_true": Array of ground truth values with shape (N, d).
        The following fields are included only when return_preds is True and a corresponding quantile argument is provided.
        "y_pred_lower": Array of lower quantile predictions with shape (N, d) returned when lower_quantile is not None.
        "y_pred_upper": Array of upper quantile predictions with shape (N, d) returned when upper_quantile is not None.

"""
    # basic checks
    if not isinstance(m, int) or m < 2:
        raise ValueError("m must be an integer >= 2.")
    if not isinstance(coverage, float) or not (0.0 < coverage < 1.0):
        raise ValueError("coverage must be in (0, 1).")
    if lower_quantile is not None:
        if not isinstance(lower_quantile, float) or not (0.0 < lower_quantile < 1.0):
            raise ValueError("lower_quantile must be a float in (0, 1).")
    if upper_quantile is not None:
        if not isinstance(upper_quantile, float) or not (0.0 < upper_quantile < 1.0):
            raise ValueError("upper_quantile must be a float in (0, 1).")
    if lower_quantile is not None and upper_quantile is not None and not (lower_quantile < upper_quantile):
        raise ValueError("Require lower_quantile < upper_quantile when both are provided.")

    # Define criterion
    criterion = multi_energy_loss

    # collect samples
    all_y_true, all_y_samples = [], []
    model.eval()
    with torch.inference_mode():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            B = x_batch.size(0)
            x_expanded = x_batch.repeat_interleave(m, dim=0)
            y_expanded = model(x_expanded)                  
            y_samples = y_expanded.reshape(B, m, -1)       

            all_y_true.append(y_batch)
            all_y_samples.append(y_samples)

    # to numpy
    y_true_np = torch.cat(all_y_true, dim=0).detach().cpu().numpy()
    y_samples_np = torch.cat(all_y_samples, dim=0).detach().cpu().numpy()

    # inverse transform if needed
    if scaler_y is not None:
        y_true_orig = y_true_np * scaler_y.scale_ + scaler_y.mean_
        scale = scaler_y.scale_.reshape(1, 1, -1)
        mean = scaler_y.mean_.reshape(1, 1, -1)
        y_samples_orig = y_samples_np * scale + mean
    else:
        y_true_orig = y_true_np
        y_samples_orig = y_samples_np

    # point predictions and metrics
    y_pred_mean = y_samples_orig.mean(axis=1)
    y_pred_median = np.median(y_samples_orig, axis=1).squeeze()
    mse = mean_squared_error(y_true_orig, y_pred_mean)
    mae_mean = mean_absolute_error(y_true_orig, y_pred_mean)

    # coverage and sharpness at central interval defined by coverage
    q_low_c = np.quantile(y_samples_orig, (1.0 - coverage) / 2.0, axis=1)
    q_high_c = np.quantile(y_samples_orig, 1.0 - (1.0 - coverage) / 2.0, axis=1)
    coverage_score = np.mean((y_true_orig >= q_low_c) & (y_true_orig <= q_high_c))
    sharpness_score = np.mean(q_high_c - q_low_c)

    # upper tail MSE based on training distribution
    upperq_mse = np.nan
    upperq_thresh = np.nan
    upperq_count = 0
    if isinstance(upperquantile_mse, float) and 0.0 < upperquantile_mse < 1.0:
        y_train_tensor = train_loader.dataset.tensors[1]
        y_train = (y_train_tensor.detach().cpu().numpy() if y_train_tensor.is_cuda
                   else y_train_tensor.detach().numpy()).reshape(-1)

        if scaler_y is not None:
            y_train_orig = scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
        else:
            y_train_orig = y_train

        q_thresh = np.quantile(y_train_orig, upperquantile_mse)
        extreme_mask = y_true_orig > q_thresh
        upperq_count = int(np.sum(extreme_mask))
        if upperq_count > 0:
            upperq_mse = mean_squared_error(y_true_orig[extreme_mask], y_pred_mean[extreme_mask])
        upperq_thresh = float(q_thresh)

    # energy loss on original scale
    y_true_tensor_orig = torch.tensor(y_true_orig, dtype=torch.float32, device=device)
    y_samples_tensor = torch.tensor(y_samples_orig, dtype=torch.float32, device=device)
    s1, s2, energy_loss = criterion(y_true_tensor_orig, y_samples_tensor)

    # assemble outputs
    result = {
        "energy_loss": float(energy_loss.item()),
        "s1": float(s1.item()),
        "s2": float(s2.item()),
        "mse": float(mse),
        "mae_mean": float(mae_mean),
        "coverage": float(coverage_score),
        "sharpness": float(sharpness_score),
        "upperquantile_mse": upperq_mse,
        "upperquantile_threshold": upperq_thresh,
        "upperquantile_count": upperq_count,
    }

    # optional predictions
    if return_preds:
        result["y_pred_mean"] = y_pred_mean
        result["y_pred_median"] = y_pred_median
        result["y_true"] = y_true_orig
        if lower_quantile is not None:
            result["y_pred_lower"] = np.quantile(y_samples_orig, lower_quantile, axis=1).squeeze()
        if upper_quantile is not None:
            result["y_pred_upper"] = np.quantile(y_samples_orig, upper_quantile, axis=1).squeeze()

    if verbose:
        print("Evaluation metrics on the original scale")
        print(f"RMSE mean: {np.sqrt(mse):.5f} | MAE mean: {mae_mean:.5f}")
        print(f"Energy evaluation loss: {energy_loss.item():.5f}")
        print(f"E(|Y - Ŷ|): {s1:.5f} | E(|Ŷ - Ŷ′|): {s2:.5f}")
        print(f"Coverage at {coverage:.3f}: {coverage_score:.5f} | sharpness: {sharpness_score:.5f}")
        if isinstance(upperquantile_mse, float) and 0.0 < upperquantile_mse < 1.0:
            if upperq_count > 0 and not np.isnan(upperq_mse):
                print(
                    f"RMSE mean on test samples above training q={upperquantile_mse:.3f} "
                    f"threshold {upperq_thresh:.5f} "
                    f"({upperq_count} observations): {np.sqrt(upperq_mse):.5f}"
                )
            else:
                print(
                    f"No test observations above training q={upperquantile_mse:.3f} "
                    f"threshold {upperq_thresh if not np.isnan(upperq_thresh) else float('nan'):.5f}. "
                    f"upperquantile_mse set to NaN."
                )

    return result


##################################################################################################################