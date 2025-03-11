import os
import numpy as np
import matplotlib.pyplot as plt
from l5kit.data import LocalDataManager
from l5kit.data.proto.road_network_pb2 import MapFragment

# 🔹 Указываем путь к данным
data_path = "/home/drama/temp_usr/motion-prediction/lyft-motion-prediction-autonomous-vehicles"
os.environ["L5KIT_DATA_FOLDER"] = data_path
dm = LocalDataManager(None)

# 🔹 Загружаем семантическую карту
semantic_map_path = dm.require("semantic_map/semantic_map.pb")
with open(semantic_map_path, "rb") as f:
    map_data = MapFragment()
    map_data.ParseFromString(f.read())

print("🔹 Количество элементов в карте:", len(map_data.elements))

# 🔹 Список всех координат
all_x, all_y = [], []

# 🔹 Извлекаем границы полос и сегменты
for element in map_data.elements:
    if hasattr(element, "lane") and hasattr(element.lane, "geo_frame"):
        origin_x = element.lane.geo_frame.origin.lng_e7 / 1e7
        origin_y = element.lane.geo_frame.origin.lat_e7 / 1e7

        if hasattr(element.lane, "left_boundary") and element.lane.left_boundary.vertex_deltas_x_cm:
            x, y = origin_x, origin_y
            for dx, dy in zip(element.lane.left_boundary.vertex_deltas_x_cm, element.lane.left_boundary.vertex_deltas_y_cm):
                x += dx / 1e5
                y += dy / 1e5
                all_x.append(x)
                all_y.append(y)

        if hasattr(element.lane, "right_boundary") and element.lane.right_boundary.vertex_deltas_x_cm:
            x, y = origin_x, origin_y
            for dx, dy in zip(element.lane.right_boundary.vertex_deltas_x_cm, element.lane.right_boundary.vertex_deltas_y_cm):
                x += dx / 1e5
                y += dy / 1e5
                all_x.append(x)
                all_y.append(y)

    elif hasattr(element, "segment") and hasattr(element.segment, "vertices"):  
        for v in element.segment.vertices:
            all_x.append(v.lng_e7 / 1e7)
            all_y.append(v.lat_e7 / 1e7)

# 🔹 Проверяем, есть ли данные
if not all_x or not all_y:
    print("❌ В карте нет данных!")
else:
    print(f"✅ Найдено {len(all_x)} точек для визуализации.")

    # 🔹 Центрируем координаты и переводим в метры
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

    all_x = [(x - center_x) * 111000 for x in all_x]  # 1 градус ≈ 111 км
    all_y = [(y - center_y) * 111000 for y in all_y]

    # 🔹 Визуализация карты
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(all_x, all_y, s=1, color="black")  # Точки карты

    ax.set_title("Семантическая карта")
    ax.set_xlabel("X (метры)")
    ax.set_ylabel("Y (метры)")
    ax.grid()
    plt.show()
