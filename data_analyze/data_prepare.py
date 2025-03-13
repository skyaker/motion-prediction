import os
import numpy as np
import torch
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.configs import load_config_data
from l5kit.rasterization import build_rasterizer
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mode = config["hardware"]["mode"]

# --- Настройки L5Kit ---
DATA_PATH = "../lyft-motion-prediction-autonomous-vehicles"
os.environ["L5KIT_DATA_FOLDER"] = DATA_PATH
cfg = load_config_data("lyft-config.yaml")  # Конфиг (можно изменить)
dm = LocalDataManager(None)

# --- Загружаем датасет (автоматическая обработка) ---
zarr_path = dm.require("scenes/train.zarr")  # Lyft L5Kit API загружает zarr
dataset = ChunkedDataset(zarr_path).open()

# --- Создаем растеризатор (карту) ---
rasterizer = build_rasterizer(cfg, dm)

# --- Готовим данные через AgentDataset (автоматический X, y) ---
agent_dataset = AgentDataset(cfg, dataset, rasterizer)

# --- Подготовка данных для сохранения ---
data_to_save = {
    "images": [sample["image"] for sample in agent_dataset],  # Карты сцен
    "target_positions": [sample["target_positions"] for sample in agent_dataset],  # Будущие координаты
    "target_availabilities": [sample["target_availabilities"] for sample in agent_dataset]  # Маска доступности
}

# --- Сохраняем подготовленные данные ---
torch.save(data_to_save, "./l5kit_dataset_fixed.pth")

print(f"✅ Данные подготовлены и сохранены! Количество примеров: {len(data_to_save['images'])}")
