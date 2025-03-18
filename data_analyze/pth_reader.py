import torch

data = torch.load("../dataprocessing/l5kit_dataset_part1.pth")

print(data.keys())

print(f"Images shape: {data['images'].shape}")  
print(f"Target positions shape: {data['target_positions'].shape}")
print(f"Target availabilities shape: {data['target_availabilities'].shape}")
