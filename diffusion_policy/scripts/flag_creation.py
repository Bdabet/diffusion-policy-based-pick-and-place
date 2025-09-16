import zarr
import numpy as np
import sys
import argparse

def add_z_rotation_indicator(zarr_path: str, overwrite: bool = True):
    # Open zarr file in read/write mode
    root = zarr.open(zarr_path, mode='a')
    data = root['data']

    # Load robot_joint_vel
    joint_vel = data['robot_joint_vel'][:]   # shape (N, 6)
    assert joint_vel.ndim == 2 and joint_vel.shape[1] == 6

    # Build indicator: first 5 == 0, last != 0
    is_zero_first5 = np.all(np.isclose(joint_vel[:, :5], 0.0), axis=1)
    is_nonzero_last = ~np.isclose(joint_vel[:, 5], 0.0)
    indicator = (is_zero_first5 & is_nonzero_last).astype(np.int8)

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
        dtype='i1',
        data=indicator,
        overwrite=overwrite
    )
    print(f"Added 'z_rotation_indicator' with shape {indicator.shape} to {zarr_path}")


def add_gripper_positive_flag(zarr_path: str, pre_seconds: float = 6.0, time_step: float = 0.1, overwrite: bool = True):
    # Open zarr file in read/write mode
    root = zarr.open(zarr_path, mode='a')
    data = root['data']

    # Load gripper_state
    gripper_state = data['gripper_state'][:].flatten()
    assert gripper_state.ndim == 1

    # Compute number of samples for the pre-close window
    window_samples = int(pre_seconds / time_step)

    # Initialize positive_flag array
    positive_flag = np.zeros_like(gripper_state, dtype=np.int8)

    # Find indices where gripper_state changes from 0 -> 1
    close_indices = np.where((gripper_state[1:] == 1) & (gripper_state[:-1] == 0))[0] + 1

    # Mark positive_flag starting 'pre_seconds' before each close
    for idx in close_indices:
        start_idx = max(0, idx - window_samples)
        positive_flag[start_idx:idx+1] = 1

    # Save into Zarr
    if 'positive_flag' in data:
        if overwrite:
            print("Overwriting existing 'positive_flag'")
            del data['positive_flag']
        else:
            raise RuntimeError("positive_flag already exists. Use overwrite=True to replace.")

    data.create_dataset(
        name='z_translation_indicator',
        shape=positive_flag.shape,
        dtype='i1',
        data=positive_flag,
        overwrite=overwrite
    )
    print(f"Added 'positive_flag' with shape {positive_flag.shape} to {zarr_path}")

# example usage -- python3 add_z_rotation_indicator.py /path/to/zarr --z_rotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add indicators to a Zarr file.")
    parser.add_argument("zarr_path", type=str, help="Path to Zarr file")
    parser.add_argument("--z_rotation", action="store_true", help="Add z_rotation_indicator")
    parser.add_argument("--gripper_flag", action="store_true", help="Add gripper positive flag")
    parser.add_argument("--pre_seconds", type=float, default=6.0, help="Seconds before gripper close (gripper_flag only)")
    parser.add_argument("--time_step", type=float, default=0.1, help="Time step per sample (gripper_flag only)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing indicators if present")

    args = parser.parse_args()

    if not args.z_rotation and not args.gripper_flag:
        print("You must specify at least one of --z_rotation or --gripper_flag")
        sys.exit(1)

    if args.z_rotation:
        add_z_rotation_indicator(args.zarr_path, overwrite=args.overwrite)

    if args.gripper_flag:
        add_gripper_positive_flag(args.zarr_path, pre_seconds=args.pre_seconds, time_step=args.time_step, overwrite=args.overwrite)
