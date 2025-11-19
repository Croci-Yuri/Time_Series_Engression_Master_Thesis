#################################################################################################################
                                     # Import libraries
#################################################################################################################

import torch

#################################################################################################################
                                     # Early Stopping Class
#################################################################################################################

class EarlyStopping:
    """
    Early stopping for loss minimization.

    Args:
        patience (int): Number of validation checks to wait after no improvement.
        delta (float): Minimum decrease in validation loss to count as improvement.
    """
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.best_model_state = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):

        # If validation loss improves, counter is reset and new best model is saved
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # If not, counter increase
        else:
            self.counter += 1

            # If counter reach patience level we stop
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

