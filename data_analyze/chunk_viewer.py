import zarr

root = zarr.open("../lyft-motion-prediction-autonomous-vehicles/scenes/sample.zarr", mode="r")

agents = root["agents"]

print(f"Chunk size: {agents.chunks}")

shape = agents.shape
chunks = agents.chunks
num_chunks = tuple(s // c for s, c in zip(shape, chunks))
print(f"Chunk num: {num_chunks}")

chunk_data = agents[:20000]
print(chunk_data[:5])
