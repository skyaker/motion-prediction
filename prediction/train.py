import torch
import torch.optim as optim
import torch.nn as nn
from dataset import load_dataset
from model import TrajectoryPredictor

dataloader = load_dataset("../dataprocessing/l5kit_dataset_part1.pth", batch_size=2)

model = TrajectoryPredictor()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

EPOCHS = 5

for epoch in range(EPOCHS):
  total_loss = 0
  for batch in dataloader:
    images, targets, masks, is_stationary = batch

    optimizer.zero_grad()
    predictions = model(images, is_stationary)

    loss = loss_fn(predictions * masks.unsqueeze(-1), targets * masks.unsqueeze(-1))

    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  print(f"Epoch {epoch+1}/{EPOCHS} - err: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "output_data/model.pth")
