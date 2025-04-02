import torch
import os
import time
torch.set_num_threads(1) # FOR LOCAL TRAIN ONLY
torch.set_num_interop_threads(1) # FOR LOCAL TRAIN ONLY
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

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
    print(f"Using {cfg['model_params']['num_modes']} modes for prediction.")

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
        future_len=cfg["model_params"]["future_num_frames"],
        num_modes=cfg["model_params"]["num_modes"]
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"], weight_decay=cfg["model_params"]["weight_decay"])

    num_epochs = cfg["model_params"]["train_epochs"]
    print(f"Start training for {num_epochs} epochs...")

    model.train()
    best_loss = float('inf')

    os.makedirs("server_output", exist_ok=True)
    log_file = open("server_output/train_log.txt", "a")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        nll_total = 0.0
        smooth_total = 0.0
        entropy_total = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            image = batch["image"].to(device, non_blocking=True)
            is_stationary = batch["is_stationary"].unsqueeze(1).float().to(device)
            targets = batch["target_positions"].to(device)
            availabilities = batch["target_availabilities"].to(device)

            optimizer.zero_grad()
            predictions, confidences = model(image, is_stationary)

            loss, nll_val, smooth_val, entropy_val = nll_loss(
                predictions, confidences, targets, availabilities,
                lambda_smooth=cfg["loss_params"]["lambda_smooth"],
                lambda_entropy=cfg["loss_params"]["lambda_entropy"],
                lambda_coverage=cfg["loss_params"]["lambda_coverage"]
            )

            nll_total += nll_val.item()
            smooth_total += smooth_val.item()
            entropy_total += entropy_val.item()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        log_file.write(f"Epoch {epoch+1}, avg_loss={avg_loss:.4f}, avg_nll={nll_total / len(dataloader):.4f}, avg_smooth={smooth_total / len(dataloader):.4f}, avg_entropy={entropy_total / len(dataloader):.4f}\n")
        log_file.flush()

        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/model_best.pt")

        print(f"\U0001f9ee Epoch {epoch+1} — avg NLL loss: {avg_loss:.4f}")

    log_file.close()

    model_path = cfg["model_params"]["model_path"]
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Модель сохранена: {model_path}")

if __name__ == "__main__":
    main()
