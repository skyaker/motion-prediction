import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryPredictor(nn.Module):
    def __init__(self, future_len=50, num_modes=3):
        super().__init__()
        self.num_modes = num_modes
        self.future_len = future_len

        # CNN Backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(25, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # → [B, 128, 1, 1]
        )

        # FC Layers
        self.fc = nn.Sequential(
            nn.Linear(128 + 1, 128),  # +1 for is_stationary
            nn.ReLU(),
            nn.Linear(128, num_modes * future_len * 2 + num_modes)  # trajectories + confidences
        )

    def forward(self, x, is_stationary):
        batch_size = x.shape[0]

        x = self.cnn(x)
        x = x.view(batch_size, -1)  # → [B, 128]

        if is_stationary.dim() == 1:
            is_stationary = is_stationary.unsqueeze(1)  # [B, 1]

        x = torch.cat([x, is_stationary], dim=1)  # [B, 129]
        x = self.fc(x)  # [B, num_modes * T * 2 + num_modes]

        # Разделение на траектории и вероятности
        total = self.num_modes * self.future_len * 2
        traj = x[:, :total].view(batch_size, self.num_modes, self.future_len, 2)
        confidences = F.softmax(x[:, total:], dim=1)  # [B, num_modes]

        return traj, confidences
