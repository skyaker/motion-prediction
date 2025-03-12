import os
import numpy as np
from l5kit.data import ChunkedDataset
from l5kit.data import LocalDataManager

data_path = "../lyft-motion-prediction-autonomous-vehicles"
os.environ["L5KIT_DATA_FOLDER"] = data_path
dm = LocalDataManager(None)

zarr_path = "scenes/train.zarr"
zarr_dataset = ChunkedDataset(dm.require(zarr_path)).open()

print("Датасет загружен")
print(f"Количество кадров: {len(zarr_dataset.frames)}")
print(f"Количество сцен: {len(zarr_dataset.scenes)}")
print(f"Количество объектов (агентов): {len(zarr_dataset.agents)}")
print(f"Количество дорожных объектов: {len(zarr_dataset.tl_faces)}")  # Светофоры

first_frame = zarr_dataset.frames[0]

print("Доступные ключи в первом кадре:", first_frame.dtype.names)
print("Временная метка:", first_frame["timestamp"])
print("Индекс агентов:", first_frame["agent_index_interval"])
print("Индекс светофоров:", first_frame["traffic_light_faces_index_interval"])
print("Координаты эго-автомобиля:", first_frame["ego_translation"])
print("Ориентация эго-автомобиля:", first_frame["ego_rotation"])

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

