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


def visualize_scene(image, history, target, predictions, confidences, output_path, raster_size, num_trajectories=3):
    fig, ax = plt.subplots(figsize=(8, 8))
    MAP_CHANNELS = [22, 23, 24]  # можно менять под любые слои
    img_rgb = image[MAP_CHANNELS, :, :].transpose(1, 2, 0)

    ax.imshow(img_rgb, origin='upper')

    def to_pixel_coords(points):
        return points[:, 0] * 4 + raster_size[0] * 0.25, points[:, 1] * 4 + raster_size[1] * 0.5

    # Предсказания (красные + оттенки)
    if predictions is not None and len(predictions) > 0:
        num_trajectories = min(num_trajectories, len(predictions))
        sorted_idx = np.argsort(-confidences)[:num_trajectories]  # по убыванию

        for j, idx in enumerate(sorted_idx):
            traj = predictions[idx]
            conf = confidences[idx]
            x, y = to_pixel_coords(traj)
            ax.plot(x, y, 's-', alpha=0.6 + 0.4 * (conf / confidences.max()), label=f'Pred #{j+1} ({conf:.2f})')

    # История (синяя)
    if history is not None:
        x, y = to_pixel_coords(history)
        ax.plot(x, y, 'o-', color='blue', label='History')

    # Истинная траектория (зелёная)
    if target is not None:
        x, y = to_pixel_coords(target)
        ax.plot(x, y, 'o--', color='red', label='Real')

    ax.legend()
    ax.set_title("All points visualization")
    ax.axis('off')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def visualize_scene_collage(image, history, target, predictions, confidences, output_path, raster_size, num_trajectories=3):
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
    zarr_path = cfg["test_data_loader"]["key"][mode]
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

                base_dir = "server_output/output_images/"
                os.makedirs(base_dir, exist_ok=True)
                collage_path = os.path.join(base_dir, f"collage_scene_{global_index}.png")  # или как у тебя называется

                # visualize_scene(
                #     image=img,
                #     history=history,
                #     target=target,
                #     predictions=preds,
                #     confidences=confs,
                #     output_path=save_path,
                #     raster_size=raster_size
                # )

                # collage_path = f"output_images/collage_scene_{global_index}.png"
                visualize_scene_collage(
                    image=img,
                    history=history,
                    target=target,
                    predictions=preds,
                    confidences=confs,
                    output_path=collage_path,
                    raster_size=raster_size
                )


if __name__ == "__main__":
    main()