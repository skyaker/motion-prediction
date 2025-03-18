import torch
import torch.nn as nn

class TrajectoryPredictor(nn.Module):
    def __init__(self):
        super(TrajectoryPredictor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(25, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(129, 128),
            nn.ReLU(),
            nn.Linear(128, 100) 
        )

    def forward(self, x, is_stationary):
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        is_stationary = is_stationary.mean(dim=1, keepdim=False).squeeze(1) 

        x = torch.cat([x, is_stationary.unsqueeze(1)], dim=1)

        x = self.fc(x) 
        return x.view(-1, 50, 2)

if __name__ == "__main__":
    model = TrajectoryPredictor()
    print(model)
