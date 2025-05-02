import torch
import torch.nn as nn
import timm

class TrajectoryPredictor(nn.Module):
    def __init__(self, input_channels=25, num_modes=3, future_len=50):
        super().__init__()
        self.num_modes = num_modes
        self.future_len = future_len

        self.backbone = timm.create_model(
            "efficientnet_b0", features_only=True, pretrained=True, in_chans=input_channels
        )
        backbone_out_channels = self.backbone.feature_info[-1]['num_chs']

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.traj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_out_channels, 256),
                nn.ReLU(),
                nn.Linear(256, future_len * 2)
            ) for _ in range(num_modes)
        ])

        self.confidence_head = nn.Sequential(
            nn.Linear(backbone_out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes)
        )

    def forward(self, x):
        feats = self.backbone(x)[-1]
        pooled = self.pool(feats).flatten(1)

        trajectories = [head(pooled).view(-1, self.future_len, 2) for head in self.traj_heads]
        traj_tensor = torch.stack(trajectories, dim=1)

        confidences = self.confidence_head(pooled)
        confidences = torch.softmax(confidences, dim=1)

        return traj_tensor, confidences
