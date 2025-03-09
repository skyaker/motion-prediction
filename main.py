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

# # ВИЗУАЛИЗАЦИЯ КАРТЫ: Folium
# map_center = [history_positions[-1, 1], history_positions[-1, 0]]
# m = folium.Map(location=map_center, zoom_start=16)
# for pos in history_positions:
#     folium.CircleMarker([pos[1], pos[0]], radius=3, color="blue").add_to(m)

# for pos in predicted_positions:
#     folium.CircleMarker([pos[1], pos[0]], radius=3, color="red").add_to(m)

# m.save("map.html")
# print("Карта сохранена! Открой файл map.html в браузере.")

# # ВИЗУАЛИЗАЦИЯ 3D ДАННЫХ LiDAR
# # Загружаем случайное облако точек для демонстрации
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.random.randn(10000, 3) * 10)

# # Визуализация облака точек
# o3d.visualization.draw_geometries([pcd])

import os
import numpy as np
from l5kit.data import ChunkedDataset
from l5kit.data import LocalDataManager

# Указываем путь к данным
data_path = "lyft-motion-prediction-autonomous-vehicles"
os.environ["L5KIT_DATA_FOLDER"] = data_path
dm = LocalDataManager(None)

# Загружаем train.zarr
zarr_path = "scenes/train.zarr"
zarr_dataset = ChunkedDataset(dm.require(zarr_path)).open()

# Вывод информации о датасете
print("Датасет успешно загружен!")
print(f"Количество кадров: {len(zarr_dataset.frames)}")
print(f"Количество сцен: {len(zarr_dataset.scenes)}")
print(f"Количество объектов (агентов): {len(zarr_dataset.agents)}")
print(f"Количество дорожных объектов: {len(zarr_dataset.tl_faces)}")  # Светофоры

# Выводим информацию о первом кадре
first_frame = zarr_dataset.frames[0]

print("Доступные ключи в первом кадре:", first_frame.dtype.names)
print("Временная метка:", first_frame["timestamp"])
print("Индекс агентов:", first_frame["agent_index_interval"])
print("Индекс светофоров:", first_frame["traffic_light_faces_index_interval"])
print("Координаты эго-автомобиля:", first_frame["ego_translation"])
print("Ориентация эго-автомобиля:", first_frame["ego_rotation"])

# Выводим первые 10 объектов (агентов)
print("Первые 10 объектов в датасете:")
for i, agent in enumerate(zarr_dataset.agents[:10]):
    print(f"Агент {i+1}:")
    print(f"Координаты: {agent['centroid']}")
    print(f"Скорость: {agent['velocity']}")
    print(f"Угол ориентации: {agent['yaw']}")
    print(f"Размер (ширина, длина): {agent['extent']}")

print("Первые 5 светофоров в сцене:")

for i, tl in enumerate(zarr_dataset.tl_faces[:5]):
    print(f"Светофор {i+1}:")
    print(f"  Face ID: {tl['face_id']}")
    print(f"  Traffic Light ID: {tl['traffic_light_id']}")
    print(f"  Статус: {tl['traffic_light_face_status']}")  # 0 - красный, 1 - желтый, 2 - зеленый

