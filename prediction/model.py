import torch
import torch.nn as nn
import timm

class TrajectoryPredictor(nn.Module):
    def __init__(self, input_channels=25, num_modes=3, future_len=50):
        super().__init__()
        self.num_modes = num_modes
        self.future_len = future_len

        self.backbone = timm.create_model(
            'efficientnet_b0', features_only=True, pretrained=False
        )

        backbone_out_channels = self.backbone.feature_info[-1]['num_chs']
        self.input_adapter = nn.Conv2d(input_channels, 3, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(backbone_out_channels + 1, 512),
            nn.ReLU(),
            nn.Linear(512, num_modes * future_len * 2 + num_modes)
        )

    def forward(self, x, is_stationary):
        x = self.input_adapter(x)
        feats = self.backbone(x)[-1]
        pooled = self.pool(feats).flatten(1)  # [B, C]

        # Добавим is_stationary как дополнительную фичу
        is_stationary = is_stationary.float().view(-1, 1)  # [B, 1]
        combined = torch.cat([pooled, is_stationary], dim=1)  # [B, C+1]

        out = self.head(combined)
        bs = x.shape[0]
        traj = out[:, :self.num_modes * self.future_len * 2]
        traj = traj.view(bs, self.num_modes, self.future_len, 2)

        confidences = out[:, -self.num_modes:]
        confidences = torch.softmax(confidences, dim=1)

        return traj, confidences