import zarr

test_data = zarr.open('../lyft-motion-prediction-autonomous-vehicles/scenes/test.zarr', mode='r')

print(len(test_data["agents"]))
print(len(test_data["frames"]))
print(len(test_data["traffic_light_faces"]))

print(test_data.tree())
