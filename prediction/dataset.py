from l5kit.dataset import AgentDataset
from torch.utils.data import Dataset
import numpy as np
import zarr
import json
from l5kit.data import MapAPI, map_api
from zarr.storage import NestedDirectoryStore

class TrajectoryDataset(Dataset):
    def __init__(self, cfg, zarr_dataset, rasterizer):
        self.agent_dataset = AgentDataset(cfg, zarr_dataset, rasterizer)

        mode = cfg["hardware"]["mode"]

        zarr_path = cfg["train_data_loader"]["key"]
        print(zarr_path)

        semantic_map_path = cfg["raster_params"]["semantic_map_key"]
        meta_path = cfg["raster_params"]["dataset_meta_key"]
        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.map_api = MapAPI(semantic_map_path, meta["world_to_ecef"])
        self.tl_helper = self.map_api

        self.zarr_root = zarr.open(NestedDirectoryStore(zarr_path), mode="r")
        self.traffic_light_faces = self.zarr_root["traffic_light_faces"]
    

    def __len__(self):
        return len(self.agent_dataset)


    def find_lane_id_by_position(self, xy):
        bounds = self.map_api.get_bounds()
        for bbox, lane_id in zip(bounds["lanes"]["bounds"], bounds["lanes"]["ids"]):
            (x_min, y_min), (x_max, y_max) = bbox
            if x_min <= xy[0] <= x_max and y_min <= xy[1] <= y_max:
                return lane_id
        return None


    def find_relevant_traffic_light(self, lane_id: str, frame_index: int) -> dict:
        """
        Поиск светофора, относящегося к данной полосе движения и активного в текущем кадре.
        Возвращает словарь с цветом, ID и прочей диагностикой.
        """
        # Получить traffic control элементы, относящиеся к полосе
        tce_ids = self.map_api.get_lane_traffic_control_ids(lane_id)
        # print("---------------------------TCE_IDS--------------------------------")
        # print(tce_ids)

        tl_ids = [tce for tce in tce_ids if self.map_api.is_traffic_light(tce)]
        # print("---------------------------TL_IDS--------------------------------")
        # print(tl_ids)

        if not tl_ids:
            return {"status": "no_tl_on_lane"}

        # Получить интервал светофоров в текущем кадре
        frame = self.zarr_root["frames"][frame_index]

        # print("---------------------------FRAME--------------------------------")
        # print(frame)

        start, end = frame["traffic_light_faces_index_interval"]

        # print("---------------------------TL-INTERVAL--------------------------------")
        # print(frame["traffic_light_faces_index_interval"])

        faces = self.zarr_root["traffic_light_faces"][start:end]

        # print("---------------------------ZARR-TL-FACES--------------------------------")
        # print(self.zarr_root["traffic_light_faces"])

        # Преобразуем активные фейсы в face_id -> status
        face_data = []

        # print(faces)

        for face in faces:
            # print("==========ENTERED==========")
            face_id = face["face_id"].decode("utf-8") if isinstance(face["face_id"], bytes) else face["face_id"]
            tl_id = face["traffic_light_id"].decode("utf-8") if isinstance(face["traffic_light_id"], bytes) else face["traffic_light_id"]
            status = face["traffic_light_face_status"]  # [0, 0, 1]
            
            # print(f"========================TLID:{tl_id}========================")

            if tl_id in tl_ids:
                color = np.argmax(status)  # 0=red, 1=yellow, 2=green
                face_data.append({
                    "face_id": face_id,
                    "traffic_light_id": tl_id,
                    "color_index": int(color),
                    "color_name": ["red", "yellow", "green"][color]
                })

        # print("---------------------------FACE-DATA---------------------------")
        # print(face_data)

        if not face_data:
            return {"status": "no_active_tl_for_lane", "lane_id": lane_id}

        selected = face_data[0]  # можно заменить на ближайший к midline
        selected["status"] = "ok"
        return selected


    def __getitem__(self, idx):
        data = self.agent_dataset[idx]

        history = data["history_positions"]
        diffs = np.linalg.norm(history[1:] - history[:-1], axis=1)
        is_stationary = float(np.mean(diffs) < 0.1)

        # velocity
        velocity = data.get("current_velocity", np.zeros(2))  # fallback
        vx, vy = velocity[0], velocity[1]

        # angle
        yaw = data["yaw"]
        sin_yaw = np.sin(yaw)
        cos_yaw = np.cos(yaw)

        # 4 additional channels: [vx, vy, sin(yaw), cos(yaw)]
        _, H, W = data["image"].shape
        extra_channels = np.ones((4, H, W), dtype=np.float32)
        extra_channels[0, :, :] *= vx
        extra_channels[1, :, :] *= vy
        extra_channels[2, :, :] *= np.sin(yaw)
        extra_channels[3, :, :] *= np.cos(yaw)

        # ------------------DEBUG------------------
        # print("image shape:", data["image"].shape)
        # print("extra shape:", extra_channels.shape)
        # -----------------------------------------

        # traffic light
        xy = data["centroid"]
        lane_id = self.find_lane_id_by_position(xy)
        
        # print("##################################################LANE ID##################################################")
        # print(f"{lane_id}")

        traffic_light_status = 4
            
        if lane_id is not None:
            # print(f"succeed---------------------------------------------------------------------")
            tl_info = self.find_relevant_traffic_light(lane_id, data["frame_index"])
            # print(tl_info)
            if tl_info["status"] == "ok":
                # print("aklsdjas;ldfjas;ldf")
                traffic_light_status = tl_info["color_index"]  # 0, 1, 2
                # print(traffic_light_status)

        # print(traffic_light_status)


        data["traffic_light_feature"] = np.float32(traffic_light_status)

        data["image"] = np.concatenate([data["image"], extra_channels], axis=0)

        return {
            "centroid": data["centroid"],
            "yaw": data["yaw"],
            "history_positions": data["history_positions"],
            "image": data["image"],
            "target_positions": data["target_positions"],
            "target_availabilities": data["target_availabilities"],
            "is_stationary": is_stationary,
            "traffic_light_status": data["traffic_light_feature"] 
        }
