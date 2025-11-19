#################################################################################################################
                                            # Import libraries
#################################################################################################################


import matplotlib.pyplot as plt
import numpy as np

#######################################################################################################################
                                # Train-validation plotting for engression-based models
#######################################################################################################################


def plot_history(history,
                 title="Engression training and validation loss",
                 ylabel="Energy Loss",
                 show_best=True,
                 ylog=False,
                 start_epoch=0):
    """
    Simplified plotter for stochastic Engression training histories.
    Args:
        history (dict): Training history containing
            'epoch' (list[int]),
            'train_loss' (list[float]),
            'val_epoch' (list[int]),
            'val_loss' (list[float]),
            and optionally 'best_epoch' (int).
        title (str): Title of the plot.
        ylabel (str): Label for the y-axis.
        show_best (bool): Whether to draw a vertical line at the best validation epoch.
        ylog (bool): Whether to use logarithmic scale on the y-axis.
        start_epoch (int): Ignore all points strictly before this epoch for visualization.
    Returns:
        ax: Matplotlib Axes object with the plot.
    """
    epochs = np.array(history["epoch"])
    train_loss = np.array(history["train_loss"])
    val_epochs = np.array(history["val_epoch"])
    val_loss = np.array(history["val_loss"])

    # Apply start_epoch cut
    mask_train = epochs >= start_epoch
    epochs, train_loss = epochs[mask_train], train_loss[mask_train]

    mask_val = val_epochs >= start_epoch
    val_epochs, val_loss = val_epochs[mask_val], val_loss[mask_val]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    ax.plot(epochs, train_loss, lw=2, label="Train")
    ax.plot(val_epochs, val_loss, lw=2, label="Validation")

    if show_best and "best_epoch" in history and history["best_epoch"] is not None:
        be = int(history["best_epoch"])
        if be >= start_epoch:
            ax.axvline(be, linestyle=":", lw=1.4, color="gray", label="Best epoch")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylog:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return ax



#######################################################################################################################
                                # Train-validation plotting for deterministic models
#######################################################################################################################

def plot_history_deterministic(history,
                               title="Training and validation loss",
                               ylabel="MSE Loss",
                               show_best=True,
                               ylog=False,
                               start_epoch=0):
    """
    Simplified plotter for deterministic training histories.
    Args:
        history (dict): Training history containing
            'epoch' (list[int]),
            'train_loss' (list[float]),
            'val_epoch' (list[int]),
            'val_loss' (list[float]),
            and optionally 'best_epoch' (int).
        title (str): Title of the plot.
        ylabel (str): Label for the y-axis.
        show_best (bool): Whether to draw a vertical line at the best validation epoch.
        ylog (bool): Whether to use logarithmic scale on the y-axis.
        start_epoch (int): Ignore all points strictly before this epoch for visualization.
    Returns:
        ax: Matplotlib Axes object with the plot.
    """
    epochs = np.array(history["epoch"])
    train_loss = np.array(history["train_loss"])
    val_epochs = np.array(history["val_epoch"])
    val_loss = np.array(history["val_loss"])

    # Apply start_epoch cut
    mask_train = epochs >= start_epoch
    epochs, train_loss = epochs[mask_train], train_loss[mask_train]

    mask_val = val_epochs >= start_epoch
    val_epochs, val_loss = val_epochs[mask_val], val_loss[mask_val]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    ax.plot(epochs, train_loss, lw=2, label="Train")
    ax.plot(val_epochs, val_loss, lw=2, label="Validation")

    if show_best and "best_epoch" in history and history["best_epoch"] is not None:
        be = int(history["best_epoch"])
        if be >= start_epoch:
            ax.axvline(be, linestyle=":", lw=1.4, color="gray", label="Best epoch")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylog:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    return ax



########################################################################################################################
                                # Q10,Q50,Q90 plotting via fan chart for stochastic models
#########################################################################################################################

def fan_chart_series(y, q10, q50, q90, time=None, ax=None, title="Fan chart",
                     start=0, length=None):
    
    """
    Plot observed y together with an 80% quantile band and the median.
    Allows zooming into a window via start and length.
    Args:
        y (array-like): Observed time series values.
        q10 (array-like): 10th percentile predictions.
        q50 (array-like): 50th percentile (median) predictions.
        q90 (array-like): 90th percentile predictions.
        time (array-like, optional): Time indices for the x-axis. If None, uses integer indices.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure and axes.
        title (str): Title of the plot.
        start (int): Starting index for the window to plot.
        length (int, optional): Length of the window to plot. If None, plots until the end of the series.
    Returns:
        ax: Matplotlib Axes object with the plot.

    """

    y   = np.asarray(y)
    q10 = np.asarray(q10)
    q50 = np.asarray(q50)
    q90 = np.asarray(q90)

    if time is None:
        time = np.arange(len(y))
    else:
        time = np.asarray(time)

    end = len(y) if length is None else start + length

    # Slice window
    y_w   = y[start:end]
    q10_w = q10[start:end]
    q50_w = q50[start:end]
    q90_w = q90[start:end]
    t_w   = time[start:end]

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    # 80% band
    ax.fill_between(t_w, q10_w, q90_w, color="#ff7f0e", alpha=0.4, label="80% interval")

    # Median
    ax.plot(t_w, q50_w, color="#d62728", linewidth=2, label="Median (50%)")

    # Observations
    ax.plot(t_w, y_w, color="gray", linewidth=1, label=r"Observed $y_t$")

    ax.set_ylabel(r"$y_t$")
    # build an informative title with the window shown
    window_note = f"  [window: {t_w[0]} → {t_w[-1]}]"
    ax.set_title(f"{title}{window_note}", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()
    return ax