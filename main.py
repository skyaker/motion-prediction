import numpy as np
import matplotlib.pyplot as plt
from l5kit.data import LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory
from l5kit.geometry import transform_points

# Укажите путь к данным Lyft Motion Prediction
data_path = "./lyft-motion-prediction"
dm = LocalDataManager(data_path)

# Загрузка конфигурации
cfg = load_config_data("./lyft-config.yaml")

# Инициализация растеризатора и датасета
rasterizer = build_rasterizer(cfg, dm)
dataset = EgoDataset(cfg, dm, rasterizer)

# Загрузка одного примера
data = dataset[0]
image = data["image"]
target_positions = data["target_positions"]
history_positions = data["history_positions"]

# Визуализация
plt.imshow(image.transpose(1, 2, 0))  # Отображение карты
draw_trajectory(np.array(target_positions), label="Target", color="green")  # Будущая траектория
draw_trajectory(np.array(history_positions), label="History", color="red")  # Историческая траектория
plt.legend()
plt.show()
