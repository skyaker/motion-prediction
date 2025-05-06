from l5kit.configs import load_config_data

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import timm


def unwrap_mode_config(cfg, mode):
    def unwrap_block(d):
        return {k: v[mode] if isinstance(v, dict) and mode in v else v for k, v in d.items()}
    cfg["raster_params"] = unwrap_block(cfg["raster_params"])
    cfg["model_params"] = unwrap_block(cfg["model_params"])
    cfg["train_data_loader"] = unwrap_block(cfg["train_data_loader"])
    cfg["test_data_loader"] = unwrap_block(cfg["test_data_loader"]["key"])
    return cfg


class TrajectoryPredictor(nn.Module):
    def __init__(self, input_channels=31, num_modes=3, future_len=50):
        cfg = load_config_data("../config/lyft-config.yaml")
        mode = cfg["hardware"]["mode"]
        cfg = unwrap_mode_config(cfg, mode)
        
        super().__init__()
        self.num_modes = num_modes
        self.future_len = future_len

        self.backbone = timm.create_model(
            cfg["model_params"]["model_architecture"], features_only=True, pretrained=True
        )
        backbone_out_channels = self.backbone.feature_info[-1]['num_chs']

        # Подгонка под RGB вход
        self.input_adapter = nn.Conv2d(input_channels, 3, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # LSTM
        self.history_lstm = nn.LSTM(input_size=2, hidden_size=32, batch_first=True)

        self.traj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_out_channels + 1 + 1 + 1 + 4 + 32 + 1, 256),
                nn.ReLU(),
                nn.Linear(256, future_len * 2)
            )
            for _ in range(num_modes)
        ])

        self.confidence_head = nn.Sequential(
            nn.Linear(backbone_out_channels + 1 + 1 + 1 + 4 + 32 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes)
        )

    def forward(self, x, is_stationary, curvature, heading_change_rate, avg_neighbor_vx, avg_neighbor_vy, avg_neighbor_heading, n_neighbors, history_velocities, trajectory_direction):
        x = self.input_adapter(x)
        feats = self.backbone(x)[-1]
        pooled = self.pool(feats).flatten(1)  # [B, C]

        lstm_out, (h_n, _) = self.history_lstm(history_velocities)
        lstm_feature = h_n[-1]  # [B, 32]

        is_stationary = is_stationary.float().view(-1, 1)
        curvature = curvature.view(-1, 1)  
        avg_neighbor_vx = avg_neighbor_vx.view(-1, 1)
        avg_neighbor_vy = avg_neighbor_vy.view(-1, 1)
        avg_neighbor_heading = avg_neighbor_heading.view(-1, 1)
        n_neighbors = n_neighbors.view(-1, 1)
        trajectory_direction = trajectory_direction.view(-1, 1)
        context = torch.cat([pooled, is_stationary, curvature, heading_change_rate, avg_neighbor_vx, avg_neighbor_vy, avg_neighbor_heading, n_neighbors, lstm_feature, trajectory_direction], dim=1)  # [B, C+1]

        # Предсказания от каждой головы
        trajectories = []
        for head in self.traj_heads:
            out = head(context).view(-1, self.future_len, 2)  # [B, T, 2]
            trajectories.append(out)

        traj_tensor = torch.stack(trajectories, dim=1)  # [B, K, T, 2]

        confidences = self.confidence_head(context)
        confidences = torch.softmax(confidences, dim=1)  # [B, K]

        return traj_tensor, confidences
