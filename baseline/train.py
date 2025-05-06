import torch
import os
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

torch.set_num_threads(1) # FOR LOCAL TRAIN ONLY
torch.set_num_interop_threads(1) # FOR LOCAL TRAIN ONLY

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer

from dataset import TrajectoryDataset
from model import TrajectoryPredictor
from metrics import nll_loss


def unwrap_mode_config(cfg, mode):
    def unwrap_block(d):
        return {k: v[mode] if isinstance(v, dict) and mode in v else v for k, v in d.items()}
    cfg["raster_params"] = unwrap_block(cfg["raster_params"])
    cfg["model_params"] = unwrap_block(cfg["model_params"])
    cfg["train_data_loader"] = unwrap_block(cfg["train_data_loader"])
    cfg["test_data_loader"] = unwrap_block(cfg["test_data_loader"]["key"])
    return cfg


def main():
    cfg = load_config_data("../config/lyft-config.yaml")
    mode = cfg["hardware"]["mode"]
    cfg = unwrap_mode_config(cfg, mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dm = LocalDataManager("../lyft-motion-prediction-autonomous-vehicles")
    zarr_dataset = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"]))
    zarr_dataset.open()

    rasterizer = build_rasterizer(cfg, dm)
    dataset = TrajectoryDataset(cfg, zarr_dataset, rasterizer)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["train_data_loader"]["batch_size"],
        shuffle=cfg["train_data_loader"]["shuffle"],
        num_workers=cfg["train_data_loader"]["num_workers"],
        pin_memory=True
    )

    model = TrajectoryPredictor(
        input_channels=25,
        future_len=cfg["model_params"]["future_num_frames"],
        num_modes=cfg["model_params"]["num_modes"]
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"], weight_decay=cfg["model_params"]["weight_decay"])
    num_epochs = cfg["model_params"]["train_epochs"]

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            image = batch["image"].to(device)
            targets = batch["target_positions"].to(device)
            availabilities = batch["target_availabilities"].to(device)

            optimizer.zero_grad()
            predictions, confidences = model(image)

            loss, _, _, _ = nll_loss(
                predictions, confidences, targets, availabilities,
                lambda_smooth=0.0,
                lambda_entropy=0.0,
                lambda_coverage=0.0
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: avg_loss = {epoch_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "baseline_model.pth")
    print("Baseline saved")

if __name__ == "__main__":
    main()
