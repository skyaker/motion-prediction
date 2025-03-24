import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset
from model import TrajectoryPredictor
from tqdm import tqdm
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer

def load_config(path="../config/lyft-config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def world_to_image_coords(points, centroid, yaw, pixel_size, raster_size):
    angle = -yaw
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    shifted = points - centroid
    rotated = shifted @ rotation_matrix.T
    pixels = rotated / np.array(pixel_size)
    image_center = np.array(raster_size) / 2
    return pixels + image_center


def visualize_scene(image, history, target, predictions, confidences, output_path, raster_size):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Raster map
    img_sum = image.sum(axis=0)  # (C, H, W) → (H, W)
    ax.imshow(img_sum, cmap='gray', origin='upper')

    # Historical points (blue)
    if history is not None:
        ax.plot(history[:, 0] * 4 + raster_size[0] * 0.25, 
                history[:, 1] * 4 + raster_size[1] * 0.5, 
                'o-', color='blue', label='History')

    # Real trajectory (green)
    if target is not None:
        ax.plot(target[:, 0] * 4 + raster_size[0] * 0.25, 
                target[:, 1] * 4 + raster_size[1] * 0.5, 
                'o--', color='green', label='Real')

    # Predicted trajectory (red)
    if predictions is not None and len(predictions) > 0:
        best_pred = predictions[0]
        ax.plot(best_pred[:, 0] * 4 + raster_size[0] * 0.25, 
                best_pred[:, 1] * 4 + raster_size[1] * 0.5, 
                'x-', color='red', label='Prediction')

    ax.legend()
    ax.set_title("All points visualization")
    ax.axis('off')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def main():
    cfg = load_config()
    mode = cfg["hardware"]["mode"]
    zarr_path = cfg["test_data_loader"]["key"]

    dm = LocalDataManager(cfg["data_path"])
    
    zd = ChunkedDataset(dm.require(zarr_path))
    zd.open()
    
    cfg["raster_params"]["raster_size"] = cfg["raster_params"]["raster_size"][mode]

    rasterizer = build_rasterizer(cfg, dm)

    raster_size = tuple(cfg["raster_params"]["raster_size"])
    pixel_size = tuple(cfg["raster_params"]["pixel_size"])
    zarr_path = cfg["test_data_loader"]["key"]
    batch_size = cfg["test_data_loader"]["batch_size"][mode]
    model_path = cfg["model_params"]["model_path"]

    device = torch.device("cpu")

    dataset = TrajectoryDataset(cfg=cfg, zarr_dataset=zd, rasterizer=rasterizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = TrajectoryPredictor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs("output_data", exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            image = batch["image"].to(device)
            is_stationary = batch["is_stationary"].to(device).float()

            output = model(image, is_stationary)
            trajectories, confidences = output
            confidences = confidences.cpu().numpy()
            predictions = trajectories.cpu().numpy()

            for i in range(image.shape[0]):
                img = image[i].cpu().numpy()
                history = batch["history_positions"][i].cpu().numpy()
                target = batch["target_positions"][i].cpu().numpy()

                preds = predictions[i]
                confs = confidences[i]

                global_index = batch_idx * image.shape[0] + i
                save_path = f"output_images/scene_{global_index}.png"
                visualize_scene(
                    image=img,
                    history=history,
                    target=target,
                    predictions=preds,
                    confidences=confs,
                    output_path=save_path,
                    raster_size=raster_size
                )


if __name__ == "__main__":
    main()