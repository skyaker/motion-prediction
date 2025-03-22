import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer

from dataset import TrajectoryDataset
from model import TrajectoryPredictor
from metrics import nll_loss

import os
from tqdm import tqdm

def unwrap_mode_config(cfg, mode):
    """Заменяет словари с weak/strong на конкретные значения"""
    def unwrap_block(d):
        return {k: v[mode] if isinstance(v, dict) and mode in v else v for k, v in d.items()}
    
    cfg["raster_params"] = unwrap_block(cfg["raster_params"])
    cfg["model_params"] = unwrap_block(cfg["model_params"])
    cfg["train_data_loader"] = unwrap_block(cfg["train_data_loader"])
    cfg["test_data_loader"] = unwrap_block(cfg["test_data_loader"])
    return cfg


def main():
    # === Подготовка ===
    cfg = load_config_data("../config/lyft-config.yaml")
    mode = cfg.get("hardware", {}).get("mode", "weak")
    cfg = unwrap_mode_config(cfg, mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === L5Kit ===
    dm = LocalDataManager("../lyft-motion-prediction-autonomous-vehicles")
    zarr_dataset = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"]))
    # zarr_dataset = ChunkedDataset(dm.require(cfg["sample_data_loader"]["key"])) # LOCAL TEMP
    zarr_dataset.open()
    rasterizer = build_rasterizer(cfg, dm)

    dataset = TrajectoryDataset(cfg, zarr_dataset, rasterizer)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["train_data_loader"]["batch_size"],
        shuffle=cfg["train_data_loader"]["shuffle"],
        num_workers=cfg["train_data_loader"]["num_workers"]
    )

    # === Модель ===
    model = TrajectoryPredictor(future_len=cfg["model_params"]["future_num_frames"], num_modes=3)
    model.to(device)

    # === Оптимизатор ===
    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"], weight_decay=cfg["model_params"]["weight_decay"])

    num_epochs = cfg["model_params"]["train_epochs"]
    print(f"Start training for {num_epochs} epochs...")

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            image = batch["image"].to(device)  # [B, 25, 224, 224]
            is_stationary = batch["is_stationary"].unsqueeze(1).float().to(device)  # [B, 1]
            targets = batch["target_positions"].to(device)  # [B, T, 2]
            availabilities = batch["target_availabilities"].to(device)  # [B, T]

            optimizer.zero_grad()
            predictions, confidences = model(image, is_stationary)  # [B, 3, T, 2], [B, 3]
            loss = nll_loss(predictions, confidences, targets, availabilities)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"🧮 Epoch {epoch+1} — avg NLL loss: {avg_loss:.4f}")

    # === Сохранение модели ===
    model_path = cfg["model_params"]["model_path"]
    model_dir = os.path.dirname(model_path)

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    torch.save(model.state_dict(), model_path)
    print(f"✅ Модель сохранена: {model_path}")

if __name__ == "__main__":
    main()
