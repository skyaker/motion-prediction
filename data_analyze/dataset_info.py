from numpy import load
import numpy as np
import zarr

relative_dataset_path = '../lyft-motion-prediction-autonomous-vehicles'

mask_data = load(f'{relative_dataset_path}/scenes/mask.npz')
test_data = zarr.open(f'{relative_dataset_path}/scenes/test.zarr', mode='r')

def mask_analyze ():
  print(f'Mask arrays:{mask_data.files}')     # all arrays

  first_key = list(mask_data.keys())[0]
  array = mask_data[first_key]

  print(f'First array length: {len(array)}')   # *len equal to agent len
  print(f'Array: {array}')                     # Bool (included/excluded from set)

  print(f"Num of included agents: {np.sum(array)}")
  print(f"Num of excluded agents: {len(array) - np.sum(array)}")

def zarr_analyze ():
  print(f'Num of agents in zar: {len(test_data["agents"])}')

  mask = mask_data[list(mask_data.keys())[0]]

  print(f"----Included agents (True): {np.sum(mask)} ({(np.sum(mask) / len(mask)) * 100:.2f}%)")
  print(f"----Excluded agents (False): {len(mask) - np.sum(mask)} ({((len(mask) - np.sum(mask)) / len(mask)) * 100:.2f}%)")

  print(f'Num of frames in zar: {len(test_data["frames"])}')
  print(f'Num of traffic lights in zar: {len(test_data["traffic_light_faces"])}')

  excluded_agents = test_data["agents"].get_orthogonal_selection((np.where(~mask)[0],))
  excluded_labels = np.argmax(excluded_agents["label_probabilities"], axis=1)

  unique_excl, counts_excl = np.unique(excluded_labels, return_counts=True)
  print(dict(zip(unique_excl, counts_excl)))

  # -----------------------------------------------------------------------------------------------------------------

  indices = np.where(mask)[0] # true elements

  filtered_agents = test_data["agents"].get_orthogonal_selection((indices,)) # get only included agents
  print(filtered_agents.shape)  # size of included

  labels = np.argmax(filtered_agents["label_probabilities"], axis=1)

  # num of every class
  unique, counts = np.unique(labels, return_counts=True)
  print(dict(zip(unique, counts)))

def main():
  print("----------------------------Mask analyze----------------------------")
  mask_analyze()
  print("----------------------------Test zarr analyze----------------------------")
  zarr_analyze()

if __name__ == "__main__":
  main()