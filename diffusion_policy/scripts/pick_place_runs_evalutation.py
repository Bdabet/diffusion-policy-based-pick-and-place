import zarr
import numpy as np
import re
import os
import cv2

# position 1: (-0.72, -0.27)
# postion 2: (-0.76, -0.45)
# position 3: (-0.33, -0.255)



def evaluate_pick_and_place(
    zarr_path,
    target_positions,
    tolerance=0.02,
    movement_tolerance=0.005,
    stationary_time=2.0,
    save_frames=True
):
    """
    Evaluate pick and place episodes and optionally save last frames of video 1.
    
    Args:
        zarr_path (str): Path to the dataset .zarr folder
        target_positions (dict): Mapping {1: (x,y), 2: (x,y), 3: (x,y)}
        tolerance (float): Allowed distance in meters from target position
        movement_tolerance (float): Max movement allowed (m) in stationary check
        stationary_time (float): Time window (s) to check robot stability
        save_frames (bool): Whether to extract and save last frames of video 1
        
    Returns:
        dict with statistics and per-episode results
    """
    # Base folders
    base_dir = os.path.dirname(zarr_path.rstrip("/"))
    video_dir = os.path.join(base_dir, "videos")
    output_dir = os.path.join(base_dir, "results")
    if save_frames:
        os.makedirs(os.path.join(output_dir, "success"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "failure"), exist_ok=True)

    # Load dataset
    root = zarr.open(zarr_path, mode='r')
    data = root['data']
    meta = root['meta']

    eef_pose = data['robot_eef_pose'][:]       # shape (N, 6)
    timestamps = data['timestamp'][:]          # shape (N,)
    episode_ends = meta['episode_ends'][:]     # indices of episode ends
    text_goals = data['current_text_goal'][:]  # shape (N,) array of strings

    # Initialize statistics
    results = []
    success_count, failure_count = 0, 0

    start_idx = 0
    for ep, end_idx in enumerate(episode_ends):
        if ep == 8:
            continue  # Skip episode 8 due to  invalid start
        # Extract trajectory for this episode
        ep_poses = eef_pose[start_idx:end_idx]
        ep_times = timestamps[start_idx:end_idx]
        ep_texts = text_goals[start_idx:end_idx]

        # Use the last goal text
        goal_text = str(ep_texts[-1])

        # Extract the target position ID from text
        match = re.search(r"position\s*\[?(\d+)\]?", goal_text.lower())
        pos_id = None
        final_x = final_y = None
        dist = None
        stationary = False
        success = False

        if match:
            pos_id = int(match.group(1))
            if pos_id in target_positions:
                # Target XY
                target_x, target_y = target_positions[pos_id]

                # Final pose
                final_pose = ep_poses[-1]
                final_x, final_y = final_pose[0], final_pose[1]

                # Distance to target
                dist = np.sqrt((final_x - target_x)**2 + (final_y - target_y)**2)

                # --- Stationary check ---
                end_time = ep_times[-1]
                start_time = end_time - stationary_time
                mask = ep_times >= start_time
                recent_poses = ep_poses[mask]

                if len(recent_poses) > 0:
                    diffs = np.linalg.norm(recent_poses[:, :2] - [final_x, final_y], axis=1)
                    stationary = np.all(diffs <= movement_tolerance)
                    # if not stationary:
                    #     print(f"episode {ep} failed due to s test")

                # Success = both target check and stationary check
                success = (dist <= tolerance) and stationary

        # Update counters
        if success:
            success_count += 1
        else:
            failure_count += 1

        # Episode duration
        duration = ep_times[-1] - ep_times[0]

        results.append({
            "episode": ep,
            "goal_text": goal_text,
            "target_position_id": pos_id,
            "final_xy": (final_x, final_y),
            "distance": dist,
            "stationary": stationary,
            "duration_sec": duration,
            "success": success
        })

        # Extract last frame from video 1
        if save_frames:
            video_path = os.path.join(video_dir, str(ep), "1.mp4")
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    out_folder = "success" if success else "failure"
                    out_path = os.path.join(output_dir, out_folder, f"ep_{ep}.png")
                    cv2.imwrite(out_path, frame)
            else:
                print(f"⚠️ Missing video for episode {ep}: {video_path}")

        # Update start index
        start_idx = end_idx

    # Summary
    summary = {
        "total_episodes": len(episode_ends),
        "successes": success_count,
        "failures": failure_count,
        "success_rate": success_count / (len(episode_ends)-1),
        "results": results
    }

    return summary




if __name__ == "__main__":
    # Example usage
    zarr_path = "/workspace/diffusion-policy/data/pick_and_place_tex_cond_varying_light/replay_buffer.zarr"
    
    # Define constants for goal positions
    target_positions = {
        1: (-0.72, -0.255),  # Example XY for position 1
        2: (-0.52, -0.255),  # Example XY for position 2
        3: (-0.33, -0.255),  # Example XY for position 3
    }
    

    tolerance = 0.2           # meters for goal check
    movement_tolerance = 0.15 # meters for stationary check
    stationary_time = 1.0      # seconds

    summary = evaluate_pick_and_place(zarr_path, target_positions, tolerance)

    print("=== Evaluation Summary ===")
    print(f"Total Episodes: {summary['total_episodes']}")
    print(f"Successes: {summary['successes']}")
    print(f"Failures: {summary['failures']}")
    print(f"Success Rate: {summary['success_rate']*100:.2f}%")

    print("results", summary['results'][3])

    for res in summary['results']:
        if res['success'] == True:
            print(f"✅ Episode {res['episode']}: Success in {res['duration_sec']:.2f}s, Distance: {res['distance']:.3f}m")

