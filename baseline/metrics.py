import torch
import torch.nn.functional as F

def compute_smoothness_loss(predictions, alpha=1.0, beta=1.0):
    # speed
    vel = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
    vel_loss = (vel ** 2).sum(-1).sum(-1).mean()

    # acceleration (jerk)
    acc = predictions[:, :, 2:, :] - 2 * predictions[:, :, 1:-1, :] + predictions[:, :, :-2, :]
    jerk_loss = (acc ** 2).sum(-1).sum(-1).mean()

    return alpha * vel_loss + beta * jerk_loss


def compute_coverage_loss(predictions):
    vel = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]
    step_len = vel.norm(dim=-1)
    traj_len = step_len.sum(dim=-1)
    avg_traj_len = traj_len.mean()
    return -avg_traj_len


def compute_length_loss(predictions, targets, availabilities):
    """
    MSE между длиной предсказанной траектории и длиной ground truth
    predictions: [B, K, T, 2]
    targets: [B, T, 2]
    availabilities: [B, T]
    """
    pred_traj = predictions[:, 0]  # используем первую (наиболее вероятную) моду
    pred_steps = pred_traj[:, 1:] - pred_traj[:, :-1]  # [B, T-1, 2]
    pred_lengths = pred_steps.norm(dim=-1).sum(dim=-1)  # [B]

    target_steps = targets[:, 1:] - targets[:, :-1]
    target_lengths = (target_steps.norm(dim=-1) * availabilities[:, 1:]).sum(dim=-1)  # [B]

    loss = F.mse_loss(pred_lengths, target_lengths)
    return loss


def nll_loss(predictions, confidences, targets, availabilities, sigma=0.5, lambda_smooth=0.5, lambda_entropy=0.01, lambda_coverage=0.05, lambda_length=0.01, cluster_centers=None, cluster_weights=None):
    """
    predictions: [B, K, T, 2]
    confidences: [B, K] (after softmax)
    targets: [B, T, 2]
    availabilities: [B, T]
    """
    B, K, T, _ = predictions.shape

    # Ошибка
    diff = predictions - targets.unsqueeze(1)  # [B, K, T, 2]
    sq_dist = (diff ** 2).sum(-1)              # [B, K, T]
    masked_error = sq_dist * availabilities.unsqueeze(1)  # [B, K, T]
    error = masked_error.sum(-1)  # [B, K]

    # Вероятностная часть
    likelihood = confidences * torch.exp(-error / (2 * sigma ** 2))  # [B, K]
    log_likelihood = torch.log(likelihood.sum(dim=1) + 1e-9)         # [B]
    nll = -log_likelihood.mean()

    # Smoothness
    smooth = compute_smoothness_loss(predictions)

    # Entropy Loss
    entropy = -(confidences * torch.log(confidences + 1e-6)).sum(dim=1).mean()

    # Trajectory length
    coverage = compute_coverage_loss(predictions)

    # Length loss
    # length_loss = compute_length_loss(predictions, targets, availabilities)

    # Clusters
    if cluster_centers is not None and cluster_weights is not None:
        delta_target = targets[:, -1] - targets[:, 0]  # [B, 2]
        distances = torch.cdist(delta_target.unsqueeze(1), cluster_centers.unsqueeze(0))  # [B, 1, K]
        nearest_cluster = distances.argmin(dim=-1).squeeze()  # [B]
        weight = cluster_weights[nearest_cluster]  # [B]
    else:
        weight = torch.ones(predictions.shape[0], device=predictions.device)

    total_loss = (
        (weight * nll).mean() + 
        lambda_smooth * smooth + 
        lambda_entropy * entropy +
        lambda_coverage * coverage
        # lambda_length * length_loss
    )
    return total_loss, nll.detach(), smooth.detach(), entropy.detach()
