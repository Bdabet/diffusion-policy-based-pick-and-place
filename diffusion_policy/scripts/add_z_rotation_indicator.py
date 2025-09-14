import zarr
import numpy as np
import sys
import os

def add_z_rotation_indicator(zarr_path: str, overwrite: bool = True):
    # Open zarr file in read/write mode
    root = zarr.open(zarr_path, mode='a')
    data = root['data']

    # Load robot_joint_vel
    joint_vel = data['robot_joint_vel'][:]   # shape (N, 6)
    assert joint_vel.ndim == 2 and joint_vel.shape[1] == 6

    # Build indicator
    # Condition: first 5 == 0, last != 0
    is_zero_first5 = np.all(np.isclose(joint_vel[:, :5], 0.0), axis=1)
    is_nonzero_last = ~np.isclose(joint_vel[:, 5], 0.0)
    indicator = (is_zero_first5 & is_nonzero_last).astype(np.int8)  # (N,)

    # Save into zarr
    if 'z_rotation_indicator' in data:
        if overwrite:
            print("Overwriting existing 'z_rotation_indicator'")
            del data['z_rotation_indicator']
        else:
            raise RuntimeError("z_rotation_indicator already exists. Use overwrite=True to replace.")

    data.create_dataset(
        name='z_rotation_indicator',
        shape=indicator.shape,
        dtype='i1',  # int8
        data=indicator,
        overwrite=overwrite
    )

    print(f"Added 'z_rotation_indicator' with shape {indicator.shape} to {zarr_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_z_rotation_indicator.py <path_to_zarr>")
        sys.exit(1)

    zarr_path = sys.argv[1]
    add_z_rotation_indicator(zarr_path)
