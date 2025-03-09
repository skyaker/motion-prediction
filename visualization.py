# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from l5kit.data import ChunkedDataset, LocalDataManager

# # Указываем путь к данным и загружаем датасет
# data_path = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"
# os.environ["L5KIT_DATA_FOLDER"] = data_path
# dm = LocalDataManager(None)

# # Загружаем train.zarr
# zarr_dataset = ChunkedDataset(dm.require("scenes/train.zarr")).open()
# print(f"✅ Датасет загружен: {len(zarr_dataset.frames)} кадров")

# # Берем первый кадр
# first_frame = zarr_dataset.frames[0]
# ego_position = first_frame["ego_translation"][:2]

# # Получаем индексы агентов в этом кадре
# agent_start, agent_end = first_frame["agent_index_interval"]
# agents_in_frame = zarr_dataset.agents[agent_start:agent_end]

# # Извлекаем координаты агентов
# agent_positions = np.array([agent["centroid"] for agent in agents_in_frame])

# # Создаем график
# plt.figure(figsize=(10, 8))
# plt.scatter(agent_positions[:, 0], agent_positions[:, 1], c="blue", label="Агенты (автомобили, пешеходы)", alpha=0.6)
# plt.scatter(ego_position[0], ego_position[1], c="red", marker="x", s=200, label="Эго-автомобиль")

# plt.xlabel("X координата")
# plt.ylabel("Y координата")
# plt.legend()
# plt.title("Визуализация сцены: агенты и эго-автомобиль")
# plt.grid()

# # Сохранение графика
# plt.savefig("scene_visualization.png")
# print("✅ График сохранен как scene_visualization.png")

# # Отображение графика в Kaggle
# plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points

# Указываем путь к данным и загружаем датасет
data_path = "/kaggle/input/lyft-motion-prediction-autonomous-vehicles"
os.environ["L5KIT_DATA_FOLDER"] = data_path
dm = LocalDataManager(None)

# Загружаем train.zarr
zarr_dataset = ChunkedDataset(dm.require("scenes/train.zarr")).open()

# Загружаем конфигурацию для растеризатора
cfg = {
    "raster_params": {
        "raster_size": [1024, 1024],  
        "pixel_size": [0.5, 0.5],  
        "ego_center": [0.25, 0.5],  
        "map_type": "py_semantic",  
        "dataset_meta_key": "meta.json",
        "semantic_map_key": "semantic_map/semantic_map.pb",
        "set_origin_to_bottom": False,
        "filter_agents_threshold": 0.5,
        "disable_traffic_light_faces": False,  
    }
}

# Создаем растеризатор
rasterizer = build_rasterizer(cfg, dm)

# Берем первый кадр
first_frame = zarr_dataset.frames[0]
ego_translation = first_frame["ego_translation"]  # Координаты эго-автомобиля
ego_rotation = first_frame["ego_rotation"]  # Ориентация эго-автомобиля

# Получаем индексы агентов
agent_start, agent_end = first_frame["agent_index_interval"]
agents_in_frame = zarr_dataset.agents[agent_start:agent_end]

# Определяем позиции агентов
agent_positions = np.array([agent["centroid"] for agent in agents_in_frame])
agent_sizes = np.array([agent["extent"][:2] for agent in agents_in_frame])

# Определяем категории агентов
is_vehicle = agent_sizes[:, 0] > 1.5  
is_pedestrian = agent_sizes[:, 0] <= 1.5  

# Преобразуем координаты эго-автомобиля в систему карты
ego_pose = np.eye(4)
ego_pose[:3, :3] = ego_rotation
ego_pose[:3, 3] = ego_translation

# Генерируем изображение карты с растеризатором
raster_image = rasterizer.rasterize(ego_pose)

# Создаем график
fig, ax = plt.subplots(figsize=(10, 8))

# Отображаем карту
ax.imshow(raster_image, extent=[ego_translation[0] - 50, ego_translation[0] + 50,
                                ego_translation[1] - 50, ego_translation[1] + 50], alpha=0.7)

# Отображаем агентов
ax.scatter(agent_positions[is_vehicle, 0], agent_positions[is_vehicle, 1], c="blue", label="Автомобили", alpha=0.6)
ax.scatter(agent_positions[is_pedestrian, 0], agent_positions[is_pedestrian, 1], c="green", label="Пешеходы", alpha=0.6)
ax.scatter(ego_translation[0], ego_translation[1], c="red", marker="x", s=200, label="Эго-автомобиль")

ax.set_xlabel("X координата")
ax.set_ylabel("Y координата")
ax.legend()
ax.set_title("Визуализация сцены с картой")
ax.grid()

plt.savefig("scene_with_map.png")
print("✅ График сохранен как scene_with_map.png")

plt.show()
