import torch
from torch.utils.data import DataLoader, TensorDataset

def load_dataset(file_path, batch_size=2):
    data = torch.load(file_path)

    X = torch.tensor(data["images"], dtype=torch.float32)
    Y = torch.tensor(data["target_positions"], dtype=torch.float32)
    mask = torch.tensor(data["target_availabilities"], dtype=torch.float32)

    is_stationary = (Y.abs().sum(dim=-1) == 0).float().unsqueeze(-1)

    dataset = TensorDataset(X, Y, mask, is_stationary)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Data loaded: {len(dataset)}")
    return dataloader

if __name__ == "__main__":
  dataloader = load_dataset("../dataprocessing/l5kit_dataset_part1.pth")
  for batch in dataloader:
    images, targets, masks, stationary = batch
    break
