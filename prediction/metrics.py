import torch
import torch.nn.functional as F

# def nll_loss(predictions, confidences, targets, availabilities, sigma=1.0):
#     """
#     Negative Log-Likelihood loss for multimodal trajectory prediction.

#     Args:
#         predictions: Tensor [B, K, T, 2]
#         confidences: Tensor [B, K]
#         targets: Tensor [B, T, 2]
#         availabilities: Tensor [B, T]
#         sigma: float, variance (default=1.0)

#     Returns:
#         scalar loss
#     """
#     B, K, T, _ = predictions.shape

#     # [B, K, T, 2] - [B, 1, T, 2] = [B, K, T, 2]
#     diff = predictions - targets.unsqueeze(1)

#     # [B, K, T]
#     sq_dist = (diff ** 2).sum(-1)

#     # Apply availability mask
#     masked_error = sq_dist * availabilities.unsqueeze(1)

#     # Sum over time
#     error = masked_error.sum(-1)  # [B, K]

#     # Likelihood term
#     likelihood = confidences * torch.exp(-error / (2 * sigma ** 2))

#     # Avoid log(0)
#     log_likelihood = torch.log(likelihood.sum(dim=1) + 1e-9)

#     # NLL
#     return -log_likelihood.mean()


def compute_smoothness_loss(predictions, alpha=1.0, beta=1.0):
    # speed
    vel = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
    vel_loss = (vel ** 2).sum(-1).sum(-1).mean()

    # acceleration
    acc = predictions[:, :, 2:, :] - 2 * predictions[:, :, 1:-1, :] + predictions[:, :, :-2, :]
    jerk_loss = (acc ** 2).sum(-1).sum(-1).mean()

    return alpha * vel_loss + beta * jerk_loss


def nll_loss(predictions, confidences, targets, availabilities, sigma=1.0, lambda_smooth=0.5):
    B, K, T, _ = predictions.shape
    diff = predictions - targets.unsqueeze(1)
    sq_dist = (diff ** 2).sum(-1)
    masked_error = sq_dist * availabilities.unsqueeze(1)
    error = masked_error.sum(-1)
    likelihood = confidences * torch.exp(-error / (2 * sigma ** 2))
    log_likelihood = torch.log(likelihood.sum(dim=1) + 1e-9)
    nll = -log_likelihood.mean()

    smooth = compute_smoothness_loss(predictions, alpha=1.0, beta=1.0)
    # print(f"[DEBUG] NLL: {nll.item():.4f}, Smooth: {smooth.item():.4f}")
    return nll + lambda_smooth * smooth, nll.detach(), smooth.detach()
