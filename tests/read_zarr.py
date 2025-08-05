import zarr
import sys
import numpy as np

zarr_store = zarr.open("/workspace/diffusion_policy/data/temp_dir/replay_buffer.zarr", mode='r')
# np.set_printoptions(threshold=sys.maxsize)


print(zarr_store.tree())

# # Pick the needed value in the Zarr store
# gripper_state = zarr_store["data"]["gripper_state"]

# # Read the data into memory as a NumPy array
# print((gripper_state[:]))

# robot_eef_pose = zarr_store["data"]["action"]

# print(robot_eef_pose[:])