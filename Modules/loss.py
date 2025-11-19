#################################################################################################################
                                            # Import libraries
#################################################################################################################

import torch

#################################################################################################################
                            # Loss energy functions with 2 samples and beta = 1
#################################################################################################################


def multi_energy_loss(y_true, y_pred):
    
    """
    Computes the energy loss with β = 1 for m predicted samples per input.

    The loss is defined as:
        loss = s1 - 0.5 * s2

    where:
        s1 = average L2 distance between each predicted sample and the true target
        s2 = average pairwise L2 distance between predicted samples (excluding diagonal)

    Args:
        y_true (torch.Tensor): shape (B, d), true targets
        y_pred (torch.Tensor): shape (B, m, d), m samples per target

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            s1: distance to ground truth
            s2: diversity between samples
            loss: final energy score

    """
    B, m, d = y_pred.shape
    if m == 2:

        # m = 2
        y1 = y_pred[:, 0, :]
        y2 = y_pred[:, 1, :]
        s1 = 0.5 * (torch.norm(y1 - y_true, dim=1).mean() + torch.norm(y2 - y_true, dim=1).mean())
        s2 = torch.norm(y1 - y2, dim=1).mean()
        return s1, s2, s1 - 0.5 * s2
    
    # General case for m > 2 with Gram-matrix path (suggested by chatgpt to improve initial use of cdist)
    else:

        # General case for m > 2
        s1 = torch.norm(y_pred - y_true.unsqueeze(1), dim=2).mean()
        y_sq = (y_pred ** 2).sum(dim=2, keepdim=True)
        dist = y_sq + y_sq.transpose(1, 2) - 2 * torch.bmm(y_pred, y_pred.transpose(1, 2))
        dist = torch.sqrt(dist.clamp(min=1e-12))
        s2 = (dist.sum(dim=(1,2)) - dist.diagonal(dim1=1,dim2=2).sum(dim=1)).div_(m*(m-1)).mean()

        return s1, s2, s1 - 0.5 * s2