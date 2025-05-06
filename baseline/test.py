import os
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset
from model import TrajectoryPredictor
from tqdm import tqdm
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation.metrics import average_displacement_error_mean, final_displacement_error_mean


def load_config(path="../config/lyft-config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


cfg = load_config()
MODE = cfg["hardware"]["mode"]
BATCH_SIZE = cfg["test_data_loader"]["batch_size"][MODE]
FUTURE_LEN = cfg["model_params"]["future_num_frames"]
NUM_MODES = cfg["model_params"]["num_modes"]
MODEL_PATH = cfg["model_params"]["model_path"]


def main():
    zarr_path = cfg["test_data_loader"]["key"][MODE]
    dm = LocalDataManager(cfg["data_path"])
    zd = ChunkedDataset(dm.require(zarr_path))
    zd.open()

    cfg["raster_params"]["raster_size"] = cfg["raster_params"]["raster_size"][MODE]
    rasterizer = build_rasterizer(cfg, dm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrajectoryPredictor(
        input_channels=25,
        future_len=FUTURE_LEN,
        num_modes=NUM_MODES
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    dataset = TrajectoryDataset(cfg=cfg, zarr_dataset=zd, rasterizer=rasterizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    ade_list = []
    fde_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            image = batch["image"].to(device)
            targets = batch["target_positions"].to(device)
            avail = batch["target_availabilities"].to(device)

            trajectories, confidences = model(image)

            trajectories = trajectories.cpu().numpy()
            confidences = confidences.cpu().numpy()
            gt = targets.cpu().numpy()
            avail = avail.cpu().numpy()

            for i in range(gt.shape[0]):
                ade = average_displacement_error_mean(gt[i], trajectories[i], confidences[i], avail[i])
                fde = final_displacement_error_mean(gt[i], trajectories[i], confidences[i], avail[i])
                ade_list.append(ade)
                fde_list.append(fde)

    avg_ade = np.mean(ade_list)
    avg_fde = np.mean(fde_list)

    model_name = cfg["model_params"]["model_architecture"]
    output_dir = os.path.join("server_output", model_name)
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"ADE: {avg_ade:.6f}\n")
        f.write(f"FDE: {avg_fde:.6f}\n")

    print(f"Saved baseline metrics to {metrics_path}")
    print(f"ADE: {avg_ade:.4f} | FDE: {avg_fde:.4f}")

if __name__ == "__main__":
    main()
