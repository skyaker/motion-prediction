import matplotlib.pyplot as plt
import os
from l5kit.configs import load_config_data
from l5kit.rasterization import build_rasterizer
from l5kit.data import LocalDataManager, ChunkedDataset
import torch
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset
import yaml

def load_config(path="../config/lyft-config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def unwrap_mode_config(cfg, mode):
    def unwrap_block(d):
        return {k: v[mode] if isinstance(v, dict) and mode in v else v for k, v in d.items()}
    
    cfg["raster_params"] = unwrap_block(cfg["raster_params"])
    cfg["model_params"] = unwrap_block(cfg["model_params"])
    cfg["train_data_loader"] = unwrap_block(cfg["train_data_loader"])
    cfg["test_data_loader"] = unwrap_block(cfg["test_data_loader"])
    return cfg


def main():
    cfg = load_config_data("../config/lyft-config.yaml")
    mode = cfg.get("hardware", {}).get("mode", "strong")
    cfg = unwrap_mode_config(cfg, mode)

    dm = LocalDataManager("../lyft-motion-prediction-autonomous-vehicles")
    zarr_dataset = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"]))
    zarr_dataset.open()
    rasterizer = build_rasterizer(cfg, dm)

    dataset = TrajectoryDataset(cfg, zarr_dataset, rasterizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    batch = next(iter(loader))
    image = batch["image"][0].numpy()  # shape: (25, H, W)

    print(f"Image shape: {image.shape}")
    print("Displaying individual channels...")

    for i in range(image.shape[0]):
        # arr = image[i]
        # print(f"Channel {i} - min: {arr.min():.4f}, max: {arr.max():.4f}, mean: {arr.mean():.4f}")

        # plt.imshow(image[i], cmap='gray', vmin=0, vmax=0.01)

        # normed = (image[i] - image[i].min()) / (image[i].ptp() + 1e-6)
        # plt.imshow(normed, cmap='gray')

        plt.imshow(image[i], cmap='plasma')

        plt.title(f"Channel {i}")
        plt.axis('off')
        plt.tight_layout()
        save_path = os.path.join("raster_channels_images", f"channel_{i}.png")
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    main()
