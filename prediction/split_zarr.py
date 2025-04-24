import os
import numpy as np
from l5kit.data import ChunkedDataset
from l5kit.configs import load_config_data


def generate_mask_manual(dataset, history_num_frames, future_num_frames, filter_agents_threshold):
    agents = dataset.agents
    frames = dataset.frames

    valid = np.zeros(len(agents), dtype=bool)

    for frame_idx, frame in enumerate(frames):
        if frame_idx - history_num_frames < 0:
            continue
        if frame_idx + future_num_frames >= len(frames):
            continue

        agent_slice = slice(*frame["agent_index_interval"])
        if agent_slice.stop <= agent_slice.start:
            continue

        agents_in_frame = agents[agent_slice]
        probs = agents_in_frame["label_probabilities"][:, [3, 12, 14]]  # классы 0, 1, 2
        valid_agents = (probs > filter_agents_threshold).any(axis=1)
        # valid_agents = agents_in_frame["label_probabilities"][:, 3] > filter_agents_threshold
        valid[agent_slice] = valid_agents

    return valid


INPUT_ZARR = "../lyft-motion-prediction-autonomous-vehicles/scenes/train.zarr"  # <- путь к полному датасету
OUTPUT_ZARR = "../lyft-motion-prediction-autonomous-vehicles/scenes/train_subset_5k.zarr"  # <- куда сохранить поднабор
CONFIG_PATH = "../config/config.yaml"

# === Full dataset load ===
src_dataset = ChunkedDataset(INPUT_ZARR)
src_dataset.open()

# === Scenes ===

# subset_size = min(600, len(src_dataset.scenes))
subset_size = 100
scene_start = src_dataset.scenes[:subset_size]["frame_index_interval"][:, 0]
scene_end = src_dataset.scenes[:subset_size]["frame_index_interval"][:, 1]
frame_idxs = np.concatenate([np.arange(start, end) for start, end in zip(scene_start, scene_end)])

agent_start = src_dataset.frames[frame_idxs]["agent_index_interval"][:, 0]
agent_end = src_dataset.frames[frame_idxs]["agent_index_interval"][:, 1]
agent_idxs = np.concatenate([np.arange(start, end) for start, end in zip(agent_start, agent_end)])

# === New .zarr slice ===
dst_dataset = ChunkedDataset(OUTPUT_ZARR)
dst_dataset.initialize(
    mode="w",
    num_scenes=subset_size,
    num_frames=len(frame_idxs),
    num_agents=len(agent_idxs),
    num_tl_faces=0
)
dst_dataset.scenes[:] = src_dataset.scenes[:subset_size]
dst_dataset.frames[:] = src_dataset.frames[frame_idxs]
dst_dataset.agents[:] = src_dataset.agents[agent_idxs]

print(f"Новый .zarr: {OUTPUT_ZARR}")

# === Mask generation ===

from l5kit.dataset.select_agents import select_agents

print("Генерация agents_mask.npy")

select_agents(
    zarr_dataset=dst_dataset,
    th_agent_prob=0.5,
    th_yaw_degree=30,
    th_extent_ratio=1.1,
    th_distance_av=50
)
#
#
# cfg = load_config_data(CONFIG_PATH)
# dst_dataset = ChunkedDataset(OUTPUT_ZARR)
# dst_dataset.open()
#
# mask = generate_mask_manual(
#     dataset=dst_dataset,
#     history_num_frames=cfg["model_params"]["history_num_frames"],
#     future_num_frames=cfg["model_params"]["future_num_frames"],
#     filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"])
#
# np.save(os.path.join(OUTPUT_ZARR, "agents_mask.npy"), mask)
# print(f"✅ Маска сохранена: {os.path.join(OUTPUT_ZARR, 'agents_mask.npy')}")
#

