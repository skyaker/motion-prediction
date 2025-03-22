from l5kit.dataset import AgentDataset
from torch.utils.data import Dataset
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, cfg, zarr_dataset, rasterizer):
        self.agent_dataset = AgentDataset(cfg, zarr_dataset, rasterizer)

    def __len__(self):
        return len(self.agent_dataset)

    def __getitem__(self, idx):
        data = self.agent_dataset[idx]

        history = data["history_positions"]
        diffs = np.linalg.norm(history[1:] - history[:-1], axis=1)
        is_stationary = float(np.mean(diffs) < 0.1) 

        return {
            "centroid": data["centroid"],
            "yaw": data["yaw"],
            "history_positions": data["history_positions"],
            "image": data["image"],
            "target_positions": data["target_positions"],
            "target_availabilities": data["target_availabilities"],
            "is_stationary": is_stationary
        }
