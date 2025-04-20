import torch
import torch.nn as nn
import timm

class TrajectoryPredictor(nn.Module):
    def __init__(self, input_channels=31, num_modes=3, future_len=50):
        super().__init__()
        self.num_modes = num_modes
        self.future_len = future_len

        # Фича-экстрактор
        self.backbone = timm.create_model(
            'efficientnet_b0', features_only=True, pretrained=False
        )
        backbone_out_channels = self.backbone.feature_info[-1]['num_chs']

        # Подгонка под RGB вход
        self.input_adapter = nn.Conv2d(input_channels, 3, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Модальные head'ы
        self.traj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_out_channels + 1 + 1 + 1 + 4, 256),
                nn.ReLU(),
                nn.Linear(256, future_len * 2)
            )
            for _ in range(num_modes)
        ])

        # Head для вероятностей
        self.confidence_head = nn.Sequential(
            nn.Linear(backbone_out_channels + 1 + 1 + 1 +4, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes)
        )

    def forward(self, x, is_stationary, curvature, heading_change_rate, avg_neighbor_vx, avg_neighbor_vy, avg_neighbor_heading, n_neighbors):
        x = self.input_adapter(x)
        feats = self.backbone(x)[-1]
        pooled = self.pool(feats).flatten(1)  # [B, C]

        is_stationary = is_stationary.float().view(-1, 1)
        curvature = curvature.view(-1, 1)  
        avg_neighbor_vx = avg_neighbor_vx.view(-1, 1)
        avg_neighbor_vy = avg_neighbor_vy.view(-1, 1)
        avg_neighbor_heading = avg_neighbor_heading.view(-1, 1)
        n_neighbors = n_neighbors.view(-1, 1)
        context = torch.cat([pooled, is_stationary, curvature, heading_change_rate, avg_neighbor_vx, avg_neighbor_vy, avg_neighbor_heading, n_neighbors], dim=1)  # [B, C+1]

        # Предсказания от каждой головы
        trajectories = []
        for head in self.traj_heads:
            out = head(context).view(-1, self.future_len, 2)  # [B, T, 2]
            trajectories.append(out)

        traj_tensor = torch.stack(trajectories, dim=1)  # [B, K, T, 2]

        # Предсказания вероятностей
        confidences = self.confidence_head(context)
        confidences = torch.softmax(confidences, dim=1)  # [B, K]

        return traj_tensor, confidences
