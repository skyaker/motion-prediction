import torch
import os
import time
import numpy as np
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

from sklearn.cluster import KMeans
from collections import Counter


def print_gpu_utilization():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"GPU utilization: {torch.cuda.utilization()} %")


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
    print(str(zarr_dataset))

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
    model_architecture = cfg["model_params"]["model_architecture"]

    optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"], weight_decay=cfg["model_params"]["weight_decay"])

    num_epochs = cfg["model_params"]["train_epochs"]
    print(f"Start training for {num_epochs} epochs...")

    model.train()
    best_loss = float('inf')

    os.makedirs("server_output", exist_ok=True)
    log_file = open("server_output/train_log.txt", "a")

    resume_training = True # переключатель
    checkpoint_path = f"./checkpoints/model_best_{model_architecture}.pt"
    start_epoch = 0

    if resume_training and os.path.exists(checkpoint_path):
        print("Train from checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
    else:
        print("Training from scratch")

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_loss = 0.0
        nll_total = 0.0
        smooth_total = 0.0
        entropy_total = 0.0

        epoch_targets = []
        epoch_predictions = []

        # TRAINING BY FRAMES ----------------------------------------------------------------

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}"):
            if torch.cuda.is_available():
                print_gpu_utilization()

            image = batch["image"].to(device, non_blocking=True)
            is_stationary = batch["is_stationary"].unsqueeze(1).float().to(device)
            targets = batch["target_positions"].to(device)
            availabilities = batch["target_availabilities"].to(device)
            curvature = batch["curvature"].to(device)
            heading_change_rate = batch["heading_change_rate"].to(device)
            avg_neighbor_vx = batch["avg_neighbor_vx"].to(device).float()
            avg_neighbor_vy = batch["avg_neighbor_vy"].to(device).float()
            avg_neighbor_heading = batch["avg_neighbor_heading"].to(device).float()
            n_neighbors = batch["n_neighbors"].to(device).float()
            history_velocities = batch["history_velocities"].to(device).float()
            trajectory_direction = batch["trajectory_direction"].to(device).float()

            optimizer.zero_grad()
            predictions, confidences = model(image, is_stationary, curvature, heading_change_rate, avg_neighbor_vx, avg_neighbor_vy, avg_neighbor_heading, n_neighbors, history_velocities, trajectory_direction)

            loss, nll_val, smooth_val, entropy_val = nll_loss(
                predictions, confidences, targets, availabilities,
                lambda_smooth=cfg["loss_params"]["lambda_smooth"],
                lambda_entropy=cfg["loss_params"]["lambda_entropy"],
                lambda_coverage=cfg["loss_params"]["lambda_coverage"],
            )

            nll_total += nll_val.item()
            smooth_total += smooth_val.item()
            entropy_total += entropy_val.item()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            epoch_targets.append(targets.detach().cpu())
            epoch_predictions.append(predictions.detach().cpu())


        # LOSS LOGGING ----------------------------------------------------------------------

        avg_loss = epoch_loss / len(dataloader)

        log_path = "./server_output/info_log.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as log_file:
            log_file.write(f"Epoch {epoch+1}, avg_loss={avg_loss:.4f}, avg_nll={nll_total / len(dataloader):.4f}, avg_smooth={smooth_total / len(dataloader):.4f}, avg_entropy={entropy_total / len(dataloader):.4f}\n")
            log_file.flush()

        # PREDICTION LENGTH LOGGING ---------------------------------------------------------

        targets_all = torch.cat(epoch_targets, dim=0)
        preds_all = torch.cat(epoch_predictions, dim=0)

        gt_deltas = targets_all[:, 1:] - targets_all[:, :-1]
        gt_lengths = torch.norm(gt_deltas, dim=-1).sum(dim=-1)

        pred_deltas = preds_all[:, 0, 1:] - preds_all[:, 0, :-1]
        pred_lengths = torch.norm(pred_deltas, dim=-1).sum(dim=-1)

        avg_gt_len = gt_lengths.mean().item()
        avg_pred_len = pred_lengths.mean().item()

        log_path = "./server_output/info_log.txt"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as log_file:
            log_file.write(f"[Epoch {epoch}] Avg GT Len: {avg_gt_len:.2f} | Pred Len: {avg_pred_len:.2f}\\n")
        print(f"[Epoch {epoch}] Avg GT Len: {avg_gt_len:.2f} | Pred Len: {avg_pred_len:.2f}")


        # MODEL, CHECKPOINT SAVE ------------------------------------------------------------

        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/model_{model_architecture}_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, checkpoint_path)

        # BEST MODEL SAVE -------------------------------------------------------------------

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, f"checkpoints/model_best_{model_architecture}.pt")

        print(f"\U0001f9ee Epoch {epoch+1} — avg NLL loss: {avg_loss:.4f}")

    log_file.close()

    model_path = f"{model_architecture}.pth"
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

if __name__ == "__main__":
    main()
