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


def visualize_multi_agent_collage(image, agents_data, output_dir, frame_index, num_trajectories=3):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"multi_scene_frame_{frame_index}.png")

    MAP_CHANNELS = [22, 23, 24]
    img_rgb = image[MAP_CHANNELS, :, :].transpose(1, 2, 0)

    def to_pixel_coords(points):
        return points[:, 0] * 4 + image.shape[2] * 0.25, points[:, 1] * 4 + image.shape[1] * 0.5

    n_agents = len(agents_data)
    fig, axs = plt.subplots(n_agents + 1, 3, figsize=(18, 5 * (n_agents + 1)))
    fig.suptitle(f"Multi-Agent Scene Collage | Frame {frame_index}", fontsize=16)

    pred_colors = ['lightblue', 'orange', 'lightgreen']
    markers = ['s-', 'P-', 'X-']

    for i, agent in enumerate(agents_data):
        sorted_idx = np.argsort(-agent["confidences"])[:num_trajectories]

        # Predictions
        for j, idx in enumerate(sorted_idx):
            x, y = to_pixel_coords(agent["predictions"][idx])
            axs[i, j].imshow(img_rgb, origin='upper')
            axs[i, j].plot(x, y, markers[j], color=pred_colors[j], label=f'Pred #{j+1} ({agent["confidences"][idx]:.2f})')
            axs[i, j].set_title(f"Agent {i} - Prediction #{j+1}", fontsize=10)
            axs[i, j].legend(fontsize=6)
            axs[i, j].axis('off')

        # History
        axs[i, 0].plot(*to_pixel_coords(agent["history"]), 'o-', color='blue', label='History')
        axs[i, 0].legend(fontsize=6)

        # Ground Truth
        axs[i, 1].plot(*to_pixel_coords(agent["target"]), 'o--', color='red', label='GT')
        axs[i, 1].legend(fontsize=6)

        # Combined
        axs[i, 2].imshow(img_rgb, origin='upper')
        for j, idx in enumerate(sorted_idx):
            x, y = to_pixel_coords(agent["predictions"][idx])
            axs[i, 2].plot(x, y, markers[j], color=pred_colors[j], label=f'Pred #{j+1}')
        axs[i, 2].plot(*to_pixel_coords(agent["history"]), 'o-', color='blue', label='History')
        axs[i, 2].plot(*to_pixel_coords(agent["target"]), 'o--', color='red', label='GT')
        axs[i, 2].set_title(f"Agent {i} - Combined", fontsize=10)
        axs[i, 2].legend(fontsize=6)
        axs[i, 2].axis('off')

    # Финальная сцена — все агенты вместе
    ax_final = axs[n_agents]
    if n_agents == 1:
        ax_final = [axs[n_agents, j] for j in range(3)]

    for ax in ax_final:
        ax.imshow(img_rgb, origin='upper')
        ax.set_title("Full Scene", fontsize=10)
        ax.axis('off')

    for i, agent in enumerate(agents_data):
        sorted_idx = np.argsort(-agent["confidences"])[:num_trajectories]
        for j, idx in enumerate(sorted_idx):
            x, y = to_pixel_coords(agent["predictions"][idx])
            ax_final[j].plot(x, y, markers[j], alpha=0.6)
        ax_final[2].plot(*to_pixel_coords(agent["history"]), 'o-', color='blue')
        ax_final[2].plot(*to_pixel_coords(agent["target"]), 'o--', color='red')

    plt.tight_layout()
    plt.savefig(output_path)
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

            trajectories, confidences = model(image, is_stationary)
            confidences = confidences.cpu().numpy()
            predictions = trajectories.cpu().numpy()

            # for i in range(image.shape[0]):
            #     img = image[i].cpu().numpy()
            #     history = batch["history_positions"][i].cpu().numpy()
            #     target = batch["target_positions"][i].cpu().numpy()
            #
            #     preds = predictions[i]  # [K, T, 2]
            #     confs = confidences[i]  # [K]
            #     global_index = batch_idx * image.shape[0] + i
            #
            #     collage_path = f"server_output/output_images/collage_scene_{global_index}.png"
            #     visualize_scene_collage(
            #         image=img,
            #         history=history,
            #         target=target,
            #         predictions=preds,
            #         confidences=confs,
            #         output_path=collage_path,
            #         raster_size=raster_size
            #     )

            # Собираем мультиагентную сцену для визуализации
            agents_data = []
            for i in range(image.shape[0]):
                img = image[i].cpu().numpy()
                history = batch["history_positions"][i].cpu().numpy()
                target = batch["target_positions"][i].cpu().numpy()
                preds = predictions[i]
                confs = confidences[i]
                agents_data.append({
                    "history": history,
                    "target": target,
                    "predictions": preds,
                    "confidences": confs
                })
            if batch_idx == 0:
                print(batch.keys())
            
            raster_size = img.shape[1:]  # (H, W)
            output_path = f"server_output/output_images"
            visualize_multi_agent_collage(img, agents_data, output_path, batch_idx)


if __name__ == "__main__":
    main()
