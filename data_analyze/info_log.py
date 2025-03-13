import os
import numpy as np
import pandas as pd
from l5kit.data import ChunkedDataset, LocalDataManager
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mode = config["hardware"]["mode"]

# --- НАСТРОЙКИ ---
DATA_PATH = "../lyft-motion-prediction-autonomous-vehicles"
OUTPUT_PATH = "logs_output"  # папка для логов
os.environ["L5KIT_DATA_FOLDER"] = DATA_PATH
dm = LocalDataManager(None)

# --- ЗАГРУЖАЕМ ДАННЫЕ ---
zarr_path = "scenes/train.zarr"
dataset = ChunkedDataset(dm.require(zarr_path)).open()

print(f"Всего кадров: {len(dataset.frames)}")

# --- НАСТРОЙКИ ЛОГИРОВАНИЯ ---
SAMPLE_RATE = config["processing"]["sample_rate"][mode]
BATCH_SIZE = config["processing"]["batch_size"][mode]
total_frames = len(dataset.frames)

# --- СОЗДАЕМ ПАПКУ ДЛЯ ЛОГОВ ---
os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- СОХРАНЯЕМ ЛОГИ ---
frame_ids = np.random.choice(total_frames, int(total_frames * SAMPLE_RATE), replace=False)  # выбираем 1%
frame_ids.sort()  # сортируем по времени

logs_txt = []
logs_parquet = []

for idx, frame_id in enumerate(frame_ids):
    frame = dataset.frames[frame_id]
    timestamp = frame["timestamp"]
    ego_pos = frame["ego_translation"][:2]
    ego_yaw = np.arctan2(frame["ego_rotation"][1, 0], frame["ego_rotation"][0, 0])

    log_entry = f"Frame ID: {frame_id}\n"
    log_entry += f"Ego Car: x={ego_pos[0]:.2f}, y={ego_pos[1]:.2f}, θ={ego_yaw:.4f}\n"

    agents = dataset.agents[frame["agent_index_interval"][0]: frame["agent_index_interval"][1]]

    agents_data = []
    for agent in agents:
        agent_x, agent_y = agent["centroid"]
        agent_vx, agent_vy = agent["velocity"]
        agent_yaw = agent["yaw"]

        log_entry += f"Agent: x={agent_x:.2f}, y={agent_y:.2f}, vx={agent_vx:.2f}, vy={agent_vy:.2f}, θ={agent_yaw:.4f}\n"
        agents_data.append({"frame_id": frame_id, "timestamp": timestamp, "x": agent_x, "y": agent_y,
                            "vx": agent_vx, "vy": agent_vy, "yaw": agent_yaw})

    logs_txt.append(log_entry)
    logs_parquet.extend(agents_data)

    # --- СОХРАНЯЕМ В TXT ---
    if (idx + 1) % BATCH_SIZE == 0 or idx == len(frame_ids) - 1:
        file_idx = idx // BATCH_SIZE + 1
        log_filename = os.path.join(OUTPUT_PATH, f"il_log_{file_idx}.txt")
        with open(log_filename, "w") as f:
            f.write("\n".join(logs_txt))
        logs_txt = []  # очищаем память
        print(f"Сохранен {log_filename}")

# --- СОХРАНЯЕМ В PARQUET ---
parquet_filename = os.path.join(OUTPUT_PATH, "il_logs.parquet")
df = pd.DataFrame(logs_parquet)
df.to_parquet(parquet_filename, engine="pyarrow", compression="snappy")
print(f"Сохранен {parquet_filename}")
