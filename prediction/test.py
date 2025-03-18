# import torch
# import matplotlib.pyplot as plt
# from model import TrajectoryPredictor
# from dataset import load_dataset
# import numpy as np

# model = TrajectoryPredictor()
# model.load_state_dict(torch.load("output_data/model.pth"))
# model.eval()

# dataloader = load_dataset("../dataprocessing/l5kit_dataset_part1.pth", batch_size=1)
# sample_image, sample_target, _ = next(iter(dataloader))

# with torch.no_grad():
#   predicted_trajectory = model(sample_image).squeeze(0).numpy()


# plt.imshow(sample_image.squeeze().permute(1, 2, 0))
# plt.scatter(sample_target[0, :, 0], sample_target[0, :, 1], color="green", label="Real")
# plt.scatter(predicted_trajectory[:, 0], predicted_trajectory[:, 1], color="red", label="Predicted")
# plt.legend()
# plt.title("Real & predicted")
# plt.show()
# ------------------------------------------------------
# fig, axes = plt.subplots(5, 5, figsize=(15, 15))
# sample_image = sample_image.squeeze()

# for i, ax in enumerate(axes.flat):
#     if i < 25: 
#         ax.imshow(sample_image[i], cmap="gray")
#         ax.set_title(f"Channel {i+1}")
#         ax.axis("off")

# plt.show()
# ---------------------------------All channels---------------------------------
# fig, axes = plt.subplots(5, 5, figsize=(15, 15))
# sample_image = sample_image.squeeze()

# for i, ax in enumerate(axes.flat):
#   if i < 25:
#     ax.imshow(sample_image[i], cmap="gray")
#     ax.set_title(f"Channel {i+1}")
#     ax.axis("off")

# plt.show()
# ---------------------------------Prediction---------------------------------

import torch
import matplotlib.pyplot as plt
import numpy as np
from model import TrajectoryPredictor
from dataset import load_dataset
import os

OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = TrajectoryPredictor()
model.load_state_dict(torch.load("output_data/model.pth"))
model.eval()

dataloader = load_dataset("../dataprocessing/l5kit_dataset_part1.pth", batch_size=1)

prev_scene_index = None
scene_counter = 0
image_num = 0

for sample_image, sample_target, masks, is_stationary in dataloader:
    if sample_target.numel() == 0:
        continue

    scene_index = 0
    timestamp = 0 
    print(f"Scene ID: {scene_index}, Timestamp: {timestamp}")

    if prev_scene_index is not None and prev_scene_index != scene_index:
        scene_counter += 1

    prev_scene_index = scene_index

    with torch.no_grad():
        predicted_trajectory = model(sample_image, is_stationary).squeeze(0).numpy()

    sample_target = sample_target.squeeze(0).numpy()
    masks = masks.squeeze(0).numpy()

    valid_points = sample_target[masks > 0]

    IMAGE_SIZE = 512
    PIXEL_SCALE = IMAGE_SIZE / 100
    center_x, center_y = IMAGE_SIZE // 2, IMAGE_SIZE // 2

    sample_target_px = valid_points * PIXEL_SCALE + np.array([center_x, center_y])
    predicted_trajectory_px = predicted_trajectory * PIXEL_SCALE + np.array([center_x, center_y])

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(sample_image.squeeze().mean(dim=0).numpy(), cmap="gray")
    axs[0].scatter(sample_target_px[:, 0], sample_target_px[:, 1], color="green", label="Real")
    axs[0].set_title(f"Real (Scene {scene_index})")
    axs[0].legend()
    axs[0].axis("off")

    axs[1].imshow(sample_image.squeeze().mean(dim=0).numpy(), cmap="gray")
    axs[1].scatter(predicted_trajectory_px[:, 0], predicted_trajectory_px[:, 1], color="red", label="Predicted")
    axs[1].set_title("Predicted")
    axs[1].legend()
    axs[1].axis("off")

    axs[2].imshow(sample_image.squeeze().mean(dim=0).numpy(), cmap="gray")
    axs[2].scatter(sample_target_px[:, 0], sample_target_px[:, 1], color="green", label="Real")
    axs[2].scatter(predicted_trajectory_px[:, 0], predicted_trajectory_px[:, 1], color="red", label="Predicted", marker='x')
    axs[2].set_title("Real & predicted")
    axs[2].legend()
    axs[2].axis("off")

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"scene_{scene_counter}_id_{image_num}.png")
    image_num += 1
    plt.savefig(output_path)
    plt.close(fig)

    print(f"Saved: {output_path}")
