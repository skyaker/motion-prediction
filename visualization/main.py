
# import numpy as np
# import cv2
# import l5kit
# import torch

# print("numpy version:", np.__version__)
# print("OpenCV version:", cv2.__version__)
# print("l5kit version:", l5kit.__version__)
# print("PyTorch version:", torch.__version__)
# print("MPS available:", torch.backends.mps.is_available())

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import folium
# import open3d as o3d
# from l5kit.data import LocalDataManager
# from l5kit.dataset import EgoDataset
# from l5kit.rasterization import build_rasterizer
# from l5kit.configs import load_config_data

# # ✅ Загрузка конфигурации Lyft Level 5
# cfg = load_config_data("config.yaml")
# dm = LocalDataManager(None)
# dataset = EgoDataset(cfg, dm, build_rasterizer(cfg, dm))

# # ✅ Выбираем первую сцену для визуализации
# scene_idx = 0
# frame = dataset[scene_idx]
# history_positions = frame["history_positions"]
# predicted_positions = np.random.randn(12, 2)  # Здесь должны быть предсказанные точки

# # 🚗 ВИЗУАЛИЗАЦИЯ 2D: Траектории
# plt.figure(figsize=(8, 6))
# plt.plot(history_positions[:, 0], history_positions[:, 1], "bo-", label="История движения")
# plt.plot(predicted_positions[:, 0], predicted_positions[:, 1], "ro--", label="Предсказанная траектория")
# plt.legend()
# plt.title("Траектории движения")
# plt.xlabel("X координата")
# plt.ylabel("Y координата")
# plt.grid(True)
# plt.savefig("trajectories.png")
# plt.show()

# # 🗺️ ВИЗУАЛИЗАЦИЯ КАРТЫ: Folium
# map_center = [history_positions[-1, 1], history_positions[-1, 0]]
# m = folium.Map(location=map_center, zoom_start=16)
# for pos in history_positions:
#     folium.CircleMarker([pos[1], pos[0]], radius=3, color="blue").add_to(m)

# for pos in predicted_positions:
#     folium.CircleMarker([pos[1], pos[0]], radius=3, color="red").add_to(m)

# m.save("map.html")
# print("🗺️ Карта сохранена! Открой файл map.html в браузере.")

# # 📌 ВИЗУАЛИЗАЦИЯ 3D ДАННЫХ LiDAR
# # Загружаем случайное облако точек для демонстрации
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.random.randn(10000, 3) * 10)

# # Визуализация облака точек
# o3d.visualization.draw_geometries([pcd])

from l5kit.data import ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.configs import load_config_data
from l5kit.rasterization import build_rasterizer
from l5kit.data import LocalDataManager
import os
import sys

# Указываем путь к данным
data_path = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"
os.environ["L5KIT_DATA_FOLDER"] = data_path
dm = LocalDataManager(None)

# Загружаем zarr-данные
zarr_dataset = ChunkedDataset(dm.require('train.zarr')).open()
print("Количество кадров в train.zarr:", len(zarr_dataset.frames))

# Принудительный вывод
sys.stdout.flush()