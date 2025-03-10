import os
import numpy as np
import matplotlib.pyplot as plt
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.rasterization import build_rasterizer
from l5kit.geometry import transform_points

# Dataset load
data_path = "/home/drama/temp_usr/motion-prediction/lyft-motion-prediction-autonomous-vehicles"
os.environ["L5KIT_DATA_FOLDER"] = data_path
dm = LocalDataManager(None)

# train.zarr ;pad
zarr_dataset = ChunkedDataset(dm.require("scenes/train.zarr")).open()

# Raster config
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
    },
    "model_params": {
        "history_num_frames": 10,
        "future_num_frames": 50,
        "model_architecture": "resnet18",
        "load_model": False,
        "model_path": "model.pth",
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "train_epochs": 10,
        "render_ego_history": True
    }
}

# Rasterizer creation
rasterizer = build_rasterizer(cfg, dm)

# Num history frames param
history_num_frames = cfg["model_params"]["history_num_frames"]

# 10 frames in history (indexes 0-9) i.e. 10th frame is current moment
first_frame_index = history_num_frames
first_frame = zarr_dataset.frames[first_frame_index]

# 
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

# Преобразуем в массив с правильным `dtype`
history_tl_faces_fixed = []
for tl_faces in history_tl_faces:
    history_tl_faces_fixed.append(np.array(tl_faces, dtype=[("face_id", "i4"), ("traffic_light_face_status", "3f4")]))
history_tl_faces = np.array(history_tl_faces_fixed)

print("Shape of traffic_light_face_status:", history_tl_faces[0]["traffic_light_face_status"].shape)

history_frames = np.array(history_frames, dtype=object)
history_agents = np.array(history_agents, dtype=object)

# Проверяем перед вызовом растеризатора
print(f"history_frames.shape: {history_frames.shape}")
print(f"history_agents.shape: {history_agents.shape}")
print(f"history_tl_faces.shape: {history_tl_faces.shape}")
print(f"history_tl_faces[0]: {history_tl_faces[0]}")
print(f"history_tl_faces[0]['traffic_light_face_status']: {history_tl_faces[0]['traffic_light_face_status']}")

# Вызов растеризатора
raster_image = rasterizer.rasterize(history_frames, history_agents, history_tl_faces)
raster_image = np.mean(raster_image, axis=-1)

# Определяем позиции агентов
agent_start, agent_end = first_frame["agent_index_interval"]
agents_in_frame = zarr_dataset.agents[agent_start:agent_end]
agent_positions = np.array([agent["centroid"] for agent in agents_in_frame])
agent_sizes = np.array([agent["extent"][:2] for agent in agents_in_frame])

# Определяем категории агентов
is_vehicle = agent_sizes[:, 0] > 1.5  
is_pedestrian = agent_sizes[:, 0] <= 1.5  

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

print(f"raster_image.shape: {raster_image.shape}")

plt.show()
