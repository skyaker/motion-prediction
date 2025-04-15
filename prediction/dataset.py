from l5kit.dataset import AgentDataset
from torch.utils.data import Dataset
import numpy as np
import yaml

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

        # velocity
        velocity = data.get("current_velocity", np.zeros(2))  # fallback
        vx, vy = velocity[0], velocity[1]

        # angle
        yaw = data["yaw"]
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)

        # acceleration
        step_time = 0.1
        velocities = (history[1:] - history[:-1]) / step_time

        accelerations = (velocities[1:] - velocities[:-1]) / step_time

        if len(accelerations) > 0:
            acc_mean = np.mean(accelerations, axis=0)
            acc_magnitude = np.linalg.norm(accelerations[-1])
        else:
            acc_mean = np.zeros(2)
            acc_magnitude = 0.0

        # curvature
        headings = np.arctan2(history[1:, 1] - history[:-1, 1],
                              history[1:, 0] - history[:-1, 0])
        heading_change = headings[1:] - headings[:-1]
        heading_change = (heading_change + np.pi) % (2 * np.pi) - np.pi
        if len(heading_change) > 0:
            curvature = np.mean(np.abs(heading_change))
        else:
            curvature = 0.0
        data["curvature"] = np.array([curvature], dtype=np.float32)

        # 6 additional channels: [vx, vy, sin(yaw), cos(yaw), ax, ay]
        _, H, W = data["image"].shape
        extra_channels = np.ones((6, H, W), dtype=np.float32)
        extra_channels[0, :, :] *= vx
        extra_channels[1, :, :] *= vy
        extra_channels[2, :, :] *= sin_yaw
        extra_channels[3, :, :] *= cos_yaw
        extra_channels[4, :, :] *= acc_mean[0]
        extra_channels[5, :, :] *= acc_mean[1]

        data["image"] = np.concatenate([data["image"], extra_channels], axis=0)

        return {
            "centroid": data["centroid"],
            "yaw": data["yaw"],
            "history_positions": data["history_positions"],
            "image": data["image"],
            "target_positions": data["target_positions"],
            "target_availabilities": data["target_availabilities"],
            "is_stationary": is_stationary,
            "curvature": data["curvature"]
        }
