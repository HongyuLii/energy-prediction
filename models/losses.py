import torch

def fluctuation_loss(pred, target, alpha=0.2):
    mse = torch.mean((pred - target)**2)

    # Slope loss (shape matching)
    pred_diff = pred[:, 1:] - pred[:, :-1]
    targ_diff = target[:, 1:] - target[:, :-1]
    slope_loss = torch.mean((pred_diff - targ_diff)**2)

    return mse + alpha * slope_loss
