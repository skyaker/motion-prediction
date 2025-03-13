import torch

dataset = torch.load("../data_analyze/l5kit_dataset.pth")

print(f"Размер датасета: {len(dataset)} примеров")

# Берем 1-й объект (агента)
sample = dataset[0]

# Проверяем, какие данные есть
print(f"Форма изображения (карта сцены): {sample['image'].shape}")  # [3, H, W]
print(f"Будущие координаты агента: {sample['target_positions'].shape}")  # [N, 2] (кадры, x, y)
print(f"Доступность будущих точек: {sample['target_availabilities'].shape}")  # [N]

# Проверяем первые 5 будущих точек агента
print(f"Будущая траектория (x, y): {sample['target_positions'][:5]}")
