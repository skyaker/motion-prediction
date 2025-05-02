from l5kit.dataset import AgentDataset
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, cfg, zarr_dataset, rasterizer):
        self.agent_dataset = AgentDataset(cfg, zarr_dataset, rasterizer)

    def __len__(self):
        return len(self.agent_dataset)

    def __getitem__(self, idx):
        data = self.agent_dataset[idx]

        return {
            "image": data["image"],
            "target_positions": data["target_positions"],
            "target_availabilities": data["target_availabilities"],
        }
