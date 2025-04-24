from l5kit.dataset import AgentDataset
from l5kit.data import filter_agents_by_distance
from torch.utils.data import Dataset
import numpy as np
import os
import yaml

class TrajectoryDataset(Dataset):
    def __init__(self, cfg, zarr_dataset, rasterizer):
        # path = "../lyft-motion-prediction-autonomous-vehicles/"
        # mask_path = os.path.join(path, cfg["train_data_loader"]["key"])
        # agents_mask = np.load(os.path.join(mask_path, "agents_mask.npz"))["mask"]

        self.agent_dataset = AgentDataset(cfg, zarr_dataset, rasterizer)
        # print(f"[DEBUG] agent_index (L5Kit): {self.agent_dataset.agent_index.shape}")


    def __len__(self):
        return len(self.agent_dataset)


    def __getitem__(self, idx):
        data = self.agent_dataset[idx]
        frame_index = data["frame_index"]

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

        curvature = np.clip(curvature * 2.0, 0.0, 1.0)
        data["curvature"] = np.array([curvature], dtype=np.float32)

        # vector change speed
        step_time = 0.1  # кадр = 10 Гц

        if history.shape[0] >= 3:
            dx = history[1:, 0] - history[:-1, 0]
            dy = history[1:, 1] - history[:-1, 1]
            headings = np.arctan2(dy, dx)

            heading_diff = headings[1:] - headings[:-1]
            heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi

            heading_change_rate = heading_diff[-1] / step_time if len(heading_diff) > 0 else 0.0
        else:
            heading_change_rate = 0.0

        heading_change_rate = np.clip(heading_change_rate / 3.14, -1.0, 1.0)
        data["heading_change_rate"] = np.array([heading_change_rate], dtype=np.float32)

        # neighbor feature
        frame = self.agent_dataset.dataset.frames[frame_index]
        start, end = frame["agent_index_interval"]
        all_agents = self.agent_dataset.dataset.agents[start:end]

        # исключаем самого себя по track_id
        other_agents = all_agents[all_agents["track_id"] != data["track_id"]]

        neighbors = filter_agents_by_distance(other_agents, data["centroid"], max_distance=15.0)
        n_neighbors = len(neighbors)

        if n_neighbors > 0:
            avg_velocity = np.mean(neighbors["velocity"], axis=0)
            headings = neighbors["yaw"]
            avg_heading = np.arctan2(np.mean(np.sin(headings)), np.mean(np.cos(headings)))
        else:
            avg_velocity = np.zeros(2, dtype=np.float32)
            avg_heading = 0.0

        data["avg_neighbor_vx"] = np.array([avg_velocity[0]], dtype=np.float32)
        data["avg_neighbor_vy"] = np.array([avg_velocity[1]], dtype=np.float32)
        data["avg_neighbor_heading"] = np.array([avg_heading], dtype=np.float32)
        data["n_neighbors"] = np.array([n_neighbors / 15.0], dtype=np.float32) 


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
            "curvature": data["curvature"],
            "heading_change_rate": data["heading_change_rate"],
            "avg_neighbor_vx": data["avg_neighbor_vx"],
            "avg_neighbor_vy": data["avg_neighbor_vy"],
            "avg_neighbor_heading": data["avg_neighbor_heading"],
            "n_neighbors": data["n_neighbors"]
        }
