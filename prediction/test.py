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


cfg = load_config()

MODE = cfg["hardware"]["mode"]
PIXEL_SIZE = tuple(cfg["raster_params"]["pixel_size"])
RASTER_SIZE = tuple(cfg["raster_params"]["raster_size"][MODE])
BATCH_SIZE = cfg["test_data_loader"]["batch_size"][MODE]
FUTURE_LEN = cfg["model_params"]["future_num_frames"]
NUM_MODES = cfg["model_params"]["num_modes"]
MODEL_PATH = cfg["model_params"]["model_path"]


def visualize_multi_agent_collage(images, agents_data, output_dir, frame_index, num_trajectories=3):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"multi_scene_frame_{frame_index}.png")

    MAP_CHANNELS = [22, 23, 24]

    def to_pixel_coords(points, centroid, yaw):
        if isinstance(centroid, torch.Tensor):
            centroid = centroid.cpu().numpy()
        if isinstance(yaw, torch.Tensor):
            yaw = yaw.item()
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()        

        shifted = points

        # Масштабирование в пиксели
        pixels = shifted / np.array(PIXEL_SIZE)

        # Сдвиг в центр изображения
        image_center = np.array([RASTER_SIZE[0] * 0.25, RASTER_SIZE[1] * 0.5])

        pixel_coords = pixels + image_center

        return pixel_coords[:, 0], pixel_coords[:, 1]

    n_agents = len(agents_data)
    fig, axs = plt.subplots(n_agents, 3, figsize=(18, 5 * (n_agents + 1)))
    fig.suptitle(f"Multi-Agent Scene Collage | Frame {frame_index}", fontsize=16)

    pred_colors = ['lightblue', 'orange', 'lightgreen']
    markers = ['s-', 'P-', 'X-']

    for i, agent in enumerate(agents_data):
        img_rgb = images[i][MAP_CHANNELS, :, :].transpose(1, 2, 0)

        sorted_idx = np.argsort(-agent["confidences"])[:num_trajectories]

        # History & target
        axs[i, 0].imshow(img_rgb, origin='upper')
        axs[i, 0].plot(*to_pixel_coords(agent["target"], agent["centroid"], agent["yaw"]), 'o-', color='red', label='Target')
        axs[i, 0].plot(*to_pixel_coords(agent["history"], agent["centroid"], agent["yaw"]), 'o-', color='blue', label='History')
        axs[i, 0].set_title(f"Agent {i} - History & Target", fontsize=10)
        axs[i, 0].legend(fontsize=6)
        axs[i, 0].axis('off')

        # Predictions
        axs[i, 1].imshow(img_rgb, origin='upper')
        for j, idx in reversed(list(enumerate(sorted_idx))):
            axs[i, 1].plot(*to_pixel_coords(agent["predictions"][idx], agent["centroid"], agent["yaw"]),
                        markers[j], color=pred_colors[j], label=f'Pred #{j+1} ({agent["confidences"][idx]:.2f})')
        axs[i, 1].set_title(f"Agent {i} - Predictions", fontsize=10) 
        axs[i, 1].legend(fontsize=6)
        axs[i, 1].axis('off')

        # All
        axs[i, 2].imshow(img_rgb, origin='upper')
        if num_trajectories > 2:
            axs[i, 2].plot(*to_pixel_coords(agent["predictions"][sorted_idx[2]], agent["centroid"], agent["yaw"]), 'X-', color='lightgreen', label=f'Pred #3 ({agent["confidences"][sorted_idx[2]]:.2f})')
        if num_trajectories > 1:
            axs[i, 2].plot(*to_pixel_coords(agent["predictions"][sorted_idx[1]], agent["centroid"], agent["yaw"]), 'P-', color='orange', label=f'Pred #2 ({agent["confidences"][sorted_idx[1]]:.2f})')
        axs[i, 2].plot(*to_pixel_coords(agent["target"], agent["centroid"], agent["yaw"]), 'o-', color='red', label='Target')
        axs[i, 2].plot(*to_pixel_coords(agent["predictions"][sorted_idx[0]], agent["centroid"], agent["yaw"]), 's-', color='lightblue', label=f'Pred #1 ({agent["confidences"][sorted_idx[0]]:.2f})')
        axs[i, 2].plot(*to_pixel_coords(agent["history"], agent["centroid"], agent["yaw"]), 'o-', color='blue', label='History')
        axs[i, 2].set_title(f"Agent {i} - Combined", fontsize=10)
        axs[i, 2].legend(fontsize=6)
        axs[i, 2].axis('off')

    for i, agent in enumerate(agents_data):
        sorted_idx = np.argsort(-agent["confidences"])[:num_trajectories]
        for j, idx in enumerate(sorted_idx):
            x, y = to_pixel_coords(agent["predictions"][idx], agent["centroid"], agent["yaw"])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    zarr_path = cfg["test_data_loader"]["key"][MODE]

    dm = LocalDataManager(cfg["data_path"])
    zd = ChunkedDataset(dm.require(zarr_path))
    zd.open()

    # l5kit parameters form requirement
    cfg["raster_params"]["raster_size"] = cfg["raster_params"]["raster_size"][MODE]

    rasterizer = build_rasterizer(cfg, dm)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TrajectoryDataset(cfg=cfg, zarr_dataset=zd, rasterizer=rasterizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TrajectoryPredictor(
        future_len=FUTURE_LEN,
        num_modes=NUM_MODES
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs("server_output/output_images", exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            image = batch["image"].to(device)
            images = [image[i].cpu().numpy() for i in range(image.shape[0])]

            is_stationary = batch["is_stationary"].to(device).float().unsqueeze(1)
            curvature = batch["curvature"].to(device)

            trajectories, confidences = model(image, is_stationary, curvature)
            confidences = confidences.cpu().numpy()
            predictions = trajectories.cpu().numpy()

            # Собираем мультиагентную сцену для визуализации
            agents_data = []
            for i in range(image.shape[0]):
                history = batch["history_positions"][i].cpu().numpy()
                target = batch["target_positions"][i].cpu().numpy()
                preds = predictions[i]
                confs = confidences[i]
                agents_data.append({
                    "history": history,
                    "target": target,
                    "predictions": preds,
                    "confidences": confs,
                    "centroid": batch["centroid"][i].cpu().numpy(),
                    "yaw": batch["yaw"][i].item()                
                })
            
            output_path = f"server_output/output_images"

            visualize_multi_agent_collage(images, agents_data, output_path, batch_idx)


if __name__ == "__main__":
    main()
