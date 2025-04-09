import os
import matplotlib
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
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

def visualize_scene_collage(image, history, target, predictions, confidences, traffic_light_status, output_path, raster_size, num_trajectories=3):
    MAP_CHANNELS = [22, 23, 24]
    img_rgb = image[MAP_CHANNELS, :, :].transpose(1, 2, 0)

    def to_pixel_coords(points):
        return points[:, 0] * 4 + raster_size[0] * 0.25, points[:, 1] * 4 + raster_size[1] * 0.5

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Trajectory Breakdown", fontsize=16)

    colors = ['lightblue', 'orange', 'lightgreen']
    markers = ['s-', 'P-', 'X-']
    sorted_idx = np.argsort(-confidences)[:num_trajectories]

    # Верхняя строка: 3 предсказания
    for j, idx in enumerate(sorted_idx):
        ax = axs[0, j]
        ax.imshow(img_rgb, origin='upper')
        x, y = to_pixel_coords(predictions[idx])
        ax.plot(x, y, markers[j], color=colors[j], label=f'Pred #{j+1} ({confidences[idx]:.2f})')
        ax.set_title(f"Prediction #{j+1}", fontsize=12)
        ax.axis('off')

    # Нижняя строка — история, цель, всё вместе
    # История
    ax = axs[1, 0]
    ax.imshow(img_rgb, origin='upper')
    if history is not None:
        x, y = to_pixel_coords(history)
        ax.plot(x, y, 'o-', color='blue', label='History')
    ax.set_title("History", fontsize=12)
    ax.axis('off')

    # Цель
    ax = axs[1, 1]
    ax.imshow(img_rgb, origin='upper')
    if target is not None:
        x, y = to_pixel_coords(target)
        ax.plot(x, y, 'o--', color='red', label='Real')
    ax.set_title("Ground Truth", fontsize=12)
    ax.axis('off')

    # Объединённая сцена
    ax = axs[1, 2]
    ax.imshow(img_rgb, origin='upper')
    for j, idx in enumerate(sorted_idx):
        x, y = to_pixel_coords(predictions[idx])
        ax.plot(x, y, markers[j], color=colors[j], label=f'Pred #{j+1} ({confidences[idx]:.2f})')
    if history is not None:
        x, y = to_pixel_coords(history)
        ax.plot(x, y, 'o-', color='blue', label='History')
    if target is not None:
        x, y = to_pixel_coords(target)
        ax.plot(x, y, 'o--', color='red', label='Real')
    ax.set_title("Combined", fontsize=12)

    status = traffic_light_status
    light = np.argmax(status)

    if light == 0:
        light_label = "🔴 RED"
    elif light == 1:
        light_label = "🟡 YELLOW"
    else:
        light_label = "🟢 GREEN"

    # print(traffic_light_status)

    # Добавим в легенду через искусственную линию
    handles, labels = ax.get_legend_handles_labels()
    extra = Line2D([0], [0], color='black', label=f"Traffic light: {light_label}")
    ax.legend(handles + [extra], labels + [f"Traffic light: {light_label}"])

    COLOR_MAP = {0: "unknown", 1: "green", 2: "yellow", 3: "red", 4: "none"}
    status_str = f"Traffic Light: {COLOR_MAP.get(int(traffic_light_status), 'N/A')}"

    fig.text(0.5, 0.02, status_str, ha='center', fontsize=12, bbox=dict(facecolor='lightgrey', alpha=0.5))

    ax.legend()
    ax.axis('off')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def main():
    cfg = load_config()
    mode = cfg["hardware"]["mode"]
    zarr_path = cfg["test_data_loader"]["key"][mode]

    dm = LocalDataManager(cfg["data_path"])
    zd = ChunkedDataset(dm.require(zarr_path))
    zd.open()

    cfg["raster_params"]["raster_size"] = cfg["raster_params"]["raster_size"][mode]
    rasterizer = build_rasterizer(cfg, dm)

    raster_size = tuple(cfg["raster_params"]["raster_size"])
    pixel_size = tuple(cfg["raster_params"]["pixel_size"])
    batch_size = cfg["test_data_loader"]["batch_size"][mode]
    model_path = cfg["model_params"]["model_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TrajectoryDataset(cfg=cfg, zarr_dataset=zd, rasterizer=rasterizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = TrajectoryPredictor(
        future_len=cfg["model_params"]["future_num_frames"],
        num_modes=cfg["model_params"]["num_modes"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs("server_output/output_images", exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            image = batch["image"].to(device)
            is_stationary = batch["is_stationary"].to(device).float().unsqueeze(1)
            traffic_light_status = batch["traffic_light_status"].unsqueeze(1).float().to(device)

            trajectories, confidences = model(image, is_stationary, traffic_light_status)
            confidences = confidences.cpu().numpy()
            predictions = trajectories.cpu().numpy()

            for i in range(image.shape[0]):
                img = image[i].cpu().numpy()
                history = batch["history_positions"][i].cpu().numpy()
                target = batch["target_positions"][i].cpu().numpy()
                tl_status = traffic_light_status[i].cpu().numpy()

                preds = predictions[i]  # [K, T, 2]
                confs = confidences[i]  # [K]
                global_index = batch_idx * image.shape[0] + i

                collage_path = f"server_output/output_images/collage_scene_{global_index}.png"
                visualize_scene_collage(
                    image=img,
                    history=history,
                    target=target,
                    predictions=preds,
                    confidences=confs,
                    traffic_light_status = tl_status,
                    output_path=collage_path,
                    raster_size=raster_size
                )

if __name__ == "__main__":
    main()
