import zarr
import numpy as np
import sys
import re
import argparse
from sentence_transformers import SentenceTransformer

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



def clean_current_text_goal(zarr_path: str, overwrite: bool = True):
    # Open zarr file in read/write mode
    root = zarr.open(zarr_path, mode='a')
    data = root['data']

    # Load current_text_goal
    if 'current_text_goal' not in data:
        raise RuntimeError("'current_text_goal' not found in Zarr file")
    text_goals = data['current_text_goal'][:]

    # Ensure dtype is string
    assert text_goals.dtype.kind in {'U', 'S'}

    # Clean each entry by removing "at position X"
    cleaned_goals = np.array(
        [re.sub(r" at positi\w*(?: \d+)?", "", entry) for entry in text_goals],
        dtype=text_goals.dtype
    )
    

    # Overwrite directly in the dataset (since shape doesn't change)
    if overwrite:
        print("Overwriting 'current_text_goal' entries in-place")
        data['current_text_goal'][:] = cleaned_goals
    else:
        # Optionally create a new dataset
        if 'cleaned_current_text_goal' in data:
            raise RuntimeError("cleaned_current_text_goal already exists.")
        data.create_dataset(
            name='cleaned_current_text_goal',
            shape=cleaned_goals.shape,
            dtype=text_goals.dtype,
            data=cleaned_goals,
            overwrite=overwrite
        )
        print(f"Created 'cleaned_current_text_goal' with shape {cleaned_goals.shape} in {zarr_path}")

def encode_cleaned_goals(zarr_path: str, cleaned_goals: np.ndarray, overwrite: bool = True):
    # Load the SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode the cleaned goals
    embeddings = model.encode(cleaned_goals, convert_to_numpy=True).astype(np.float32)

    # Open zarr again
    root = zarr.open(zarr_path, mode='a')
    data = root['data']

    # Save embeddings
    
    if overwrite:
            print("Overwriting existing 'cleaned_encoded_current_text_goal'")
            # del data['encoded_current_text_goal']
            data.create_dataset(
            name='encoded_current_text_goal',
            shape=embeddings.shape,
            dtype='f4',
            data=embeddings,
            overwrite=overwrite)
            print(f"Added 'encoded_current_text_goal' with shape {embeddings.shape} to {zarr_path}")
    elif 'cleaned_encoded_current_text_goal' in data:
        raise RuntimeError("cleaned_encoded_current_text_goal already exists.")
    else:
        data.create_dataset(
        name='cleaned_ecoded_current_text_goal',
        shape=embeddings.shape,
        dtype='f4',
        data=embeddings,
        overwrite=overwrite)

        print(f"Added 'cleaned_encoded_current_text_goal' with shape {embeddings.shape} to {zarr_path}")

    # data.create_dataset(
    #     name='encoded_current_text_goal',
    #     shape=embeddings.shape,
    #     dtype='f4',
    #     data=embeddings,
    #     overwrite=overwrite
    # )
    


# example usage -- python3 zarr_files_postprocessing.py /path/to/zarr --z_rotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add indicators to a Zarr file.")
    parser.add_argument("zarr_path", type=str, help="Path to Zarr file")
    parser.add_argument("--z_rotation", action="store_true", help="Add z_rotation_indicator")
    parser.add_argument("--gripper_flag", action="store_true", help="Add gripper positive flag")
    parser.add_argument("--pre_seconds", type=float, default=6.0, help="Seconds before gripper close (gripper_flag only)")
    parser.add_argument("--time_step", type=float, default=0.1, help="Time step per sample (gripper_flag only)")
    parser.add_argument("--clean_text", action="store_true", help="Clean current_text_goal by removing 'at position X'")
    parser.add_argument("--encode_cleaned", action="store_true", help="Encode the cleaned text goals")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite current_text_goal in place")

    args = parser.parse_args()


    if not args.z_rotation and not args.gripper_flag and not args.clean_text and not args.encode_cleaned:
        print("You must specify at least one of --z_rotation or --gripper_flag or --clean_text or --encode_cleaned")
        sys.exit(1)
    

    if args.encode_cleaned:
        # Reload from dataset if --clean_text wasn’t run
        root = zarr.open(args.zarr_path, mode='r')
        data = root['data']
        if 'cleaned_current_text_goal' not in data:
            raise RuntimeError("No cleaned_current_text_goal found. Run with --clean_text first or use --overwrite.")
        cleaned_goals = data['cleaned_current_text_goal'][:]
        encode_cleaned_goals(args.zarr_path, cleaned_goals, overwrite=args.overwrite)
        
    if args.clean_text:
        clean_current_text_goal(args.zarr_path, overwrite=args.overwrite)

    if args.z_rotation:
        add_z_rotation_indicator(args.zarr_path, overwrite=args.overwrite)

    if args.gripper_flag:
        add_gripper_positive_flag(args.zarr_path, pre_seconds=args.pre_seconds, time_step=args.time_step, overwrite=args.overwrite)
