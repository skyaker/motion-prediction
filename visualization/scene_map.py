import os
import numpy as np
import matplotlib.pyplot as plt
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points
import yaml

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

mode = config["hardware"]["mode"]

# Dataset load
data_path = config["paths"]["data_path"]
os.environ["L5KIT_DATA_FOLDER"] = data_path
dm = LocalDataManager(None)

# train.zarr ;pad
zarr_dataset = ChunkedDataset(dm.require("scenes/train.zarr")).open()

# Raster config
cfg = {
    "raster_params": {
        "raster_size": config["raster_params"]["raster_size"][mode],  
        "pixel_size": config["raster_params"]["pixel_size"][mode],  
        "ego_center": config["raster_params"]["ego_center"],  
        "map_type": "py_semantic",  
        "dataset_meta_key": "meta.json",
        "semantic_map_key": "semantic_map/semantic_map.pb",
        "set_origin_to_bottom": False,
        "filter_agents_threshold": 0.5,
        "disable_traffic_light_faces": False,  
    },
    "model_params": {
        "history_num_frames": config["model_params"]["history_num_frames"],
        "future_num_frames": config["model_params"]["future_num_frames"],
        "model_architecture": "resnet18",
        "load_model": False,
        "model_path": "model.pth",
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "train_epochs": config["model_params"]["train_epochs"][mode],
        "render_ego_history": True
    }
}

# Rasterizer creation
rasterizer = build_rasterizer(cfg, dm)

# Num history frames param
history_num_frames = cfg["model_params"]["history_num_frames"]

# 10 frames in history (indexes 0-9) i.e. 10th frame is current moment
scene_index = 6000

scene = zarr_dataset.scenes[scene_index]
first_frame_index = scene["frame_index_interval"][0] + history_num_frames

first_frame = zarr_dataset.frames[first_frame_index]

# Location and angle of 
ego_translation = first_frame["ego_translation"]
ego_rotation = first_frame["ego_rotation"]

# Собираем историю кадров, агентов и светофоров
history_frames = []
history_agents = []
history_tl_faces = []

for i in range(history_num_frames):
    frame = zarr_dataset.frames[first_frame_index - i]
    history_frames.append(frame)

    # Получаем агентов из текущего кадра
    agent_start, agent_end = frame["agent_index_interval"]
    agents = zarr_dataset.agents[agent_start:agent_end]
    history_agents.append(agents)

    # Получаем светофоры из текущего кадра
    tl_start, tl_end = frame["traffic_light_faces_index_interval"]
    if tl_start != tl_end:
        tl_faces = zarr_dataset.tl_faces[tl_start:tl_end]
    else:
        tl_faces = np.array([(-1, np.zeros(3, dtype=np.float32))], 
                            dtype=[('face_id', 'i4'), ('traffic_light_face_status', '3f4')])
    history_tl_faces.append(tl_faces)

history_tl_faces_fixed = []
for tl_faces in history_tl_faces:
    fixed_faces = []
    for face in tl_faces:
        try:
            face_id = int(face["face_id"])
        except (ValueError, TypeError):
            face_id = -1

        fixed_faces.append((face_id, face["traffic_light_face_status"]))

    fixed_faces = np.array(fixed_faces, dtype=[("face_id", "i4"), ("traffic_light_face_status", "3f4")])
    history_tl_faces_fixed.append(fixed_faces)

history_tl_faces = history_tl_faces_fixed

print("Shape of traffic_light_face_status:", history_tl_faces[0]["traffic_light_face_status"].shape)

history_frames = np.array(history_frames, dtype=object)
history_agents = np.array(history_agents, dtype=object)

raster_image = rasterizer.rasterize(history_frames, history_agents, history_tl_faces)
raster_image = np.mean(raster_image, axis=-1)

agent_start, agent_end = first_frame["agent_index_interval"]
agents_in_frame = zarr_dataset.agents[agent_start:agent_end]
agent_positions = np.array([agent["centroid"] for agent in agents_in_frame])

agent_sizes = np.array([agent["extent"][:2] for agent in agents_in_frame])

agent_labels = np.argmax([agent["label_probabilities"] for agent in agents_in_frame], axis=1)

unique_labels = np.unique(agent_labels)
print(f"Уникальные метки агентов в сцене: {unique_labels}")

# Определяем категории
is_vehicle = agent_labels == 3  # Автомобили
is_pedestrian = agent_labels == 14  # Пешеходы
is_cyclist = agent_labels == 12  # Велосипедисты

# Создаем график
fig, ax = plt.subplots(figsize=(10, 8))

# Отображаем карту
raster_h, raster_w = raster_image.shape
pixel_size_x, pixel_size_y = cfg["raster_params"]["pixel_size"]

# Размеры карты в метрах
raster_w_m = raster_w * pixel_size_x
raster_h_m = raster_h * pixel_size_y

# Преобразуем координаты агентов из глобальной системы в систему # Создаём корректную 4x4 матрицу трансформации
pose_in_world = np.eye(4, dtype=np.float32)  # Единичная матрица 4x4
pose_in_world[:3, :3] = ego_rotation.astype(np.float32)  # Матрица поворота (преобразуем в float32)
pose_in_world[:3, 3] = ego_translation.astype(np.float32)  # Вектор трансляции

# Инвертируем матрицу (из глобальных координат в координаты эго)
raster_from_world = np.linalg.inv(pose_in_world)  # Теперь не ломается!

# Пересчет координат эго-автомобиля в локальной системе
ego_xy_local = transform_points(np.array([[ego_translation[0], ego_translation[1], 0]]), raster_from_world)[:, :2][0]

# Пересчитываем границы extent карты (теперь они корректны)
map_x_min = ego_xy_local[0] - (cfg["raster_params"]["raster_size"][0] * cfg["raster_params"]["pixel_size"][0] / 2)
map_x_max = ego_xy_local[0] + (cfg["raster_params"]["raster_size"][0] * cfg["raster_params"]["pixel_size"][0] / 2)
map_y_min = ego_xy_local[1] - (cfg["raster_params"]["raster_size"][1] * cfg["raster_params"]["pixel_size"][1] / 2)
map_y_max = ego_xy_local[1] + (cfg["raster_params"]["raster_size"][1] * cfg["raster_params"]["pixel_size"][1] / 2)

print(f"Обновленный extent: X[{map_x_min}, {map_x_max}], Y[{map_y_min}, {map_y_max}]")

print(f"Размеры карты:")
print(f"X: {map_x_min} → {map_x_max}")
print(f"Y: {map_y_min} → {map_y_max}")

with open("../data_analyze/logs_output/sm_map_extent.log", "w") as f:
    f.write(f"Границы карты:\n")
    f.write(f"X: {map_x_min} → {map_x_max}\n")
    f.write(f"Y: {map_y_min} → {map_y_max}\n")

print("Лог с границами карты сохранён в sm_map_extent.log")

print(f"Границы карты (extent): X[{map_x_min}, {map_x_max}], Y[{map_y_min}, {map_y_max}]")
print(f"Эго-автомобиль: X = {ego_translation[0]}, Y = {ego_translation[1]}")

ax.imshow(raster_image, extent=[map_x_min, map_x_max, map_y_min, map_y_max], origin='lower', alpha=0.7)
ax.set_aspect('equal')  

agent_positions_hom = np.ones((agent_positions.shape[0], 3), dtype=np.float32)
agent_positions_hom[:, :2] = agent_positions  

agent_positions_ego = transform_points(agent_positions_hom, raster_from_world)[:, :2]

ego_x, ego_y = ego_xy_local

print(f"Ego-авто после трансформации: X = {ego_x}, Y = {ego_y}")

agent_positions_ego = transform_points(agent_positions_hom, raster_from_world)[:, :2]
with open("../data_analyze/logs_output/sm_agent_positions.log", "w") as f:

    f.write("Агент (X, Y, категория, скорость, направление)\n")
    for agent, label in zip(agents_in_frame, agent_labels):
        x, y = agent["centroid"]
        speed = np.linalg.norm(agent["velocity"])
        yaw = agent["yaw"]
        
        category = "Неизвестно"
        if label == 1:
            category = "Автомобиль"
        elif label == 2:
            category = "Пешеход"
        elif label == 3:
            category = "Велосипедист"
        f.write(f"{x}, {y}, {category}, {speed:.2f}, {yaw:.2f}\n")

print("Лог с инфой об агентах agent_positions.log")

# Отображаем агентов
ax.scatter(agent_positions_ego[is_vehicle, 0], agent_positions_ego[is_vehicle, 1], c="blue", label="Автомобили", alpha=0.6)
ax.scatter(agent_positions_ego[is_pedestrian, 0], agent_positions_ego[is_pedestrian, 1], c="green", label="Пешеходы", alpha=0.6)
ax.scatter(agent_positions_ego[is_cyclist, 0], agent_positions_ego[is_cyclist, 1], c="purple", label="Велосипедисты", alpha=0.6)
ax.scatter(ego_x, ego_y, c="red", marker="x", s=200, label="Эго-автомобиль")

ax.set_xlabel("X координата")
ax.set_ylabel("Y координата")
ax.legend()
ax.set_title("Визуализация сцены с картой")
ax.grid()

plt.savefig("images/scene_with_map.png")

print(f"Всего сцен в датасете: {len(zarr_dataset.scenes)}")
print(f"raster_image.shape: {raster_image.shape}")

plt.show()
