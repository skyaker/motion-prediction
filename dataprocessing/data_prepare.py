import os
import numpy as np
import torch
import yaml
import time
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import AgentDataset
from l5kit.configs import load_config_data
from l5kit.rasterization import build_rasterizer

# --- Загружаем конфиг ---
with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

mode = config["hardware"]["mode"]  # Выбираем "weak" или "strong"

# --- Настройки L5Kit ---
DATA_PATH = config["paths"]["data_path"]
os.environ["L5KIT_DATA_FOLDER"] = DATA_PATH
cfg = load_config_data("../config/config.yaml")  # Конфиг L5Kit
dm = LocalDataManager(None)

# --- Параметры режима (weak/strong) ---
BATCH_SIZE = config["processing"]["batch_size"][mode]  # 1000 или 10 000
SAMPLE_RATE = config["processing"]["sample_rate"][mode]  # 0.005 или 0.01
RASTER_SIZE = config["raster_params"]["raster_size"][mode]  # Размер растеризации
PIXEL_SIZE = tuple(config["raster_params"]["pixel_size"][mode]) 

cfg["raster_params"]["raster_size"] = RASTER_SIZE
cfg["raster_params"]["pixel_size"] = PIXEL_SIZE
cfg["processing"]["batch_size"] = BATCH_SIZE
cfg["processing"]["sample_rate"] = SAMPLE_RATE

# --- Загружаем датасет ---
zarr_path = dm.require(config["data_loaders"]["sample"]["key"])
dataset = ChunkedDataset(zarr_path).open()

# --- Создаем растеризатор (карту) ---
rasterizer = build_rasterizer(cfg, dm)

# --- Готовим `AgentDataset` (автоматическое создание X, y) ---
agent_dataset = AgentDataset(cfg, dataset, rasterizer)

LOG_FILE = "data_analyze/dp_log.txt"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_message(message):
    """Выводит сообщение и записывает в лог-файл"""
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    log_entry = f"{timestamp} {message}"
    print(log_entry)
    with open(LOG_FILE, "a") as log:
        log.write(log_entry + "\n")

print(f"✅ Датасет загружен: {len(agent_dataset)} примеров.")

# --- ОБРАБОТКА ДАННЫХ ПО БАТЧАМ ---
output_path = config["paths"]["output_data"]
os.makedirs(output_path, exist_ok=True)

batch_count = 0
X_list, Y_list, Mask_list = [], [], []

start_time = time.time()

for idx, sample in enumerate(agent_dataset):
    # Применяем sample rate (чтобы не перегружать данные)
    if np.random.rand() > SAMPLE_RATE:
        continue

    X_list.append(sample["image"])  # Карта сцены (входные данные)
    Y_list.append(sample["target_positions"])  # Будущие координаты (выход)
    Mask_list.append(sample["target_availabilities"])  # Маска доступности

    print(idx)

    if idx % 1000 == 0:
        elapsed_time = time.time() - start_time
        log_message(f"🟢 Обработано {idx} примеров... Время: {elapsed_time:.2f} сек")

    # Когда набираем BATCH_SIZE примеров, сохраняем их
    if len(X_list) >= BATCH_SIZE:
        batch_count += 1
        data_to_save = {
            "images": np.array(X_list, dtype=np.float32),
            "target_positions": np.array(Y_list, dtype=np.float32),
            "target_availabilities": np.array(Mask_list, dtype=np.float32)
        }
        save_path = os.path.join(output_path, f"l5kit_dataset_part{batch_count}.pth")
        torch.save(data_to_save, save_path)

        print(f"✅ Сохранен батч {batch_count} ({len(X_list)} примеров) -> {save_path}")

        # Очищаем списки для следующей партии
        X_list, Y_list, Mask_list = [], [], []

# Если остались данные, сохраняем последний файл
if X_list:
    batch_count += 1
    data_to_save = {
        "images": np.array(X_list, dtype=np.float32),
        "target_positions": np.array(Y_list, dtype=np.float32),
        "target_availabilities": np.array(Mask_list, dtype=np.float32)
    }
    save_path = os.path.join(output_path, f"l5kit_dataset_part{batch_count}.pth")
    torch.save(data_to_save, save_path)
    print(f"✅ Сохранен финальный батч {batch_count} ({len(X_list)} примеров) -> {save_path}")

print("🎉 Данные готовы к обучению!")
