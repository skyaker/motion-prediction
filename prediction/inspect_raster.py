import matplotlib.pyplot as plt
import os
from l5kit.configs import load_config_data
from l5kit.rasterization import build_rasterizer
from l5kit.data import LocalDataManager, ChunkedDataset
import torch
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset
import yaml

def load_config(path="../config/lyft-config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def unwrap_mode_config(cfg, mode):
    def unwrap_block(d):
        return {k: v[mode] if isinstance(v, dict) and mode in v else v for k, v in d.items()}
    
    cfg["raster_params"] = unwrap_block(cfg["raster_params"])
    cfg["model_params"] = unwrap_block(cfg["model_params"])
    cfg["train_data_loader"] = unwrap_block(cfg["train_data_loader"])
    cfg["test_data_loader"] = unwrap_block(cfg["test_data_loader"])
    return cfg

def main():
    cfg = load_config_data("../config/lyft-config.yaml")
    mode = cfg.get("hardware", {}).get("mode", "weak")
    cfg = unwrap_mode_config(cfg, mode)

    dm = LocalDataManager("../lyft-motion-prediction-autonomous-vehicles")
    dataset = ChunkedDataset(dm.require(cfg["train_data_loader"]["key"]))
    dataset.open()

    rasterizer = build_rasterizer(cfg, dm)

    print("Rasterizer type:", type(rasterizer).__name__)
    print("Number of channels:", rasterizer.num_channels())

    if hasattr(rasterizer, 'filter_agents_threshold'):
        print("Agents filter threshold:", rasterizer.filter_agents_threshold)

    if hasattr(rasterizer, 'history_num_frames'):
        print("History frames (ego):", rasterizer.history_num_frames)

    if hasattr(rasterizer, 'map_api'):
        print("Semantic map layers available:")
        for layer in rasterizer.map_api.get_layer_names():
            print(" -", layer)

if __name__ == "__main__":
    main()