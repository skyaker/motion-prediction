import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from torch.utils.data import DataLoader
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.rasterization import build_rasterizer
from dataset import TrajectoryDataset
from model import TrajectoryPredictor
from metrics import nll_loss
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import json
import os
import numpy as np
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


def visualize_scene(rasterizer, agent_batch, predictions, confidences, save_path, scene_id):
    """
    Отображает сцену с картой и агентами.
    Использует L5Kit `to_rgb()` для генерации правильного растра.
    """
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ['r', 'g', 'm']

    for i, agent_data in enumerate(agent_batch):
        # Получаем изображение сцены вокруг агента (с реальной картой)
        img = agent_data["image"]  # ← из TrajectoryDataset
        
        # img_vis = img.mean(axis=0)  # [112, 112]
        # ax.imshow(img_vis, cmap='viridis')  # или 'gray'

        flat = img.reshape(25, -1).T  # shape [12544, 25]

        pca = PCA(n_components=3)
        img_rgb = pca.fit_transform(flat)  # [12544, 3]

        # Вернуть обратно в [112, 112, 3]
        img_rgb = img_rgb.reshape(112, 112, 3)
        img_rgb -= img_rgb.min()
        img_rgb /= img_rgb.max()  # нормализуем до [0, 1]

        ax.imshow(img_rgb)


        # Центроид в пикселях — центр изображения
        center = np.array(img.shape[:2])[::-1] / 2

        # История (жёлтые точки)
        history = agent_data["history_positions"]
        ax.plot(history[:, 0] + center[0], history[:, 1] + center[1], 'yo', markersize=2)

        # Реальная будущая траектория (синяя)
        target = agent_data["target_positions"]
        ax.plot(target[:, 0] + center[0], target[:, 1] + center[1], 'b-', linewidth=1)

        # Предсказанные траектории (красная, зелёная, фиолетовая)
        for k in range(predictions[i].shape[0]):
            traj = predictions[i][k]
            ax.plot(traj[:, 0] + center[0], traj[:, 1] + center[1],
                    color=colors[k % len(colors)], linestyle='--', linewidth=1,
                    alpha=float(confidences[i][k]))

    ax.set_title(f"Scene {scene_id}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    cfg = load_config_data("../config/lyft-config.yaml")
    mode = cfg.get("hardware", {}).get("mode", "weak")
    cfg = unwrap_mode_config(cfg, mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = LocalDataManager("../lyft-motion-prediction-autonomous-vehicles")
    zarr_dataset = ChunkedDataset(dm.require(cfg["test_data_loader"]["key"]))
    zarr_dataset.open()
    rasterizer = build_rasterizer(cfg, dm)

    dataset = TrajectoryDataset(cfg, zarr_dataset, rasterizer)
    dataloader = DataLoader(dataset, batch_size=cfg["test_data_loader"]["batch_size"],
                            shuffle=False, num_workers=cfg["test_data_loader"]["num_workers"])

    model = TrajectoryPredictor(future_len=cfg["model_params"]["future_num_frames"], num_modes=3)
    model.load_state_dict(torch.load(cfg["model_params"]["model_path"], map_location=device))
    model.to(device)
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_confs = []
    agent_data = []

    output_dir = cfg["paths"]["output_data"]
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Testing")):
            image = batch["image"].to(device)
            is_stationary = batch["is_stationary"].unsqueeze(1).float().to(device)
            targets = batch["target_positions"].to(device)
            avail = batch["target_availabilities"].to(device)

            preds, confs = model(image, is_stationary)
            loss = nll_loss(preds, confs, targets, avail)
            total_loss += loss.item()

            # Сохраняем первые 3 примера как JSON + PNG
            if i < 3:
                agents_info = []
                for j in range(image.shape[0]):
                    agents_info.append({
                        "centroid": batch["centroid"][j].numpy(),
                        "yaw": batch["yaw"][j].numpy(),
                        "extent": np.array([4.0, 1.8, 1.7]),
                        "image": batch["image"][j].cpu().numpy(),
                        "history_positions": batch["history_positions"][j].numpy(),
                        "target_positions": batch["target_positions"][j].numpy()
                    })

                visualize_scene(
                    rasterizer,
                    agents_info,
                    preds.cpu().numpy(),
                    confs.cpu().numpy(),
                    save_path=os.path.join(output_dir, f"vis_{i}.png"),
                    scene_id=i
                )

    avg_loss = total_loss / len(dataloader)
    print(f"📉 Average NLL Loss on test set: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
