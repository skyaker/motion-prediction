with open('../lyft-motion-prediction-autonomous-vehicles/aerial_map/nearmap_images/lyft312.tfw', 'r') as f:
    content = f.readlines()
    for i, line in enumerate(content):
        print(f"Line {i+1}: {line.strip()}")
