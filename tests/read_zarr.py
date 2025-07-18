import zarr
import numpy as np

zarr_store = zarr.open("/workspace/temp_dir/replay_buffer.zarr", mode='r')


print(zarr_store.tree())

# Pick the temperature in the Zarr store
gripper_state = zarr_store["data"]["gripper_state"]

# Read the data into memory as a NumPy array
print(np.shape(gripper_state[:]))

robot_eef_pose = zarr_store["data"]["robot_eef_pose"]

print(np.shape(robot_eef_pose[:]))