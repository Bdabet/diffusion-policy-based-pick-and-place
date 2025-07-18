"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env import RealEnv
# from diffusion_policy.real_world.spacemouse_shared_memory_modified import Spacemouse
from diffusion_policy.real_world.xbox_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)


OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=120, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(input, output, robot_ip, match_dataset, match_episode,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency):




    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
    
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    print("Loaded workspace from", ckpt_path)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        print("Using device:", device)
        policy.eval().to(device)
        # print("Policy loaded:", policy)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    elif 'robomimic' in cfg.name:
        # BCRNN model
        policy: BaseImagePolicy
        policy = workspace.model

        device = torch.device('cuda')
        policy.eval().to(device)

        # BCRNN always has action horizon of 1
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        delta_action = cfg.task.dataset.get('delta_action', False)

    elif 'ibc' in cfg.name:
        policy: BaseImagePolicy
        policy = workspace.model
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096

        device = torch.device('cuda')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = 1
        delta_action = cfg.task.dataset.get('delta_action', False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
             Spacemouse(shm_manager=shm_manager) as sm, \
             RealEnv(
            output_dir=output, 
            robot_ip=robot_ip, 
            frequency=frequency,
            n_obs_steps=n_obs_steps,
            obs_image_resolution=obs_res,
            obs_float32=True,
            init_joints=init_joints,
            enable_multi_cam_vis=True,
            record_raw_video=True,
            # number of threads per camera view for video recording (H.264)
            thread_per_video=3,
            # video recording quality, lower is better (but slower).
            video_crf=21,
            shm_manager=shm_manager) as env:
            cv2.setNumThreads(1)

            # Should be the same as demo
            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            print("Waiting for realsense")
            time.sleep(1.0)

            print("Warming up policy inference")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 2
                del result

            print('Ready!')
            
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                
                
                state = env.get_robot_state()
                # print(state)
                target_pose = state['TargetTCPPose']
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    exit_while_loop = False
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    # obs = env.get_obs()

                    # visualize
                    # episode_id = env.replay_buffer.n_episodes
                    # # vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                    # match_episode_id = episode_id
                    # if match_episode is not None:
                    #     match_episode_id = match_episode
                    # if match_episode_id in episode_first_frame_map:
                    #     match_img = episode_first_frame_map[match_episode_id]
                    #     ih, iw, _ = match_img.shape
                    #     oh, ow, _ = vis_img.shape
                    #     tf = get_image_transform(
                    #         input_res=(iw, ih), 
                    #         output_res=(ow, oh), 
                    #         bgr_to_rgb=False)
                    #     match_img = tf(match_img).astype(np.float32) / 255
                    #     vis_img = np.minimum(vis_img, match_img)

                    # text = f'Episode: {episode_id}'
                    # cv2.putText(
                    #     vis_img,
                    #     text,
                    #     (10,20),
                    #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    #     fontScale=0.5,
                    #     thickness=1,
                    #     color=(255,255,255)
                    # )
                    # cv2.imshow('default', vis_img[...,::-1])

                    # create blank black image to be able to read inputs
                    # img = np.zeros((300, 300, 3), dtype=np.uint8)
                    # print("image", img)

                    # cv2.imshow("Window", img)
                    # print("image shown")#


                    press_events = key_counter.get_press_events()
                    # print(f"Press events: {press_events}")

                    for key_stroke in press_events:
                    
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            exit_while_loop = True
                            break


                    precise_wait(t_sample)
                    # print("Waited for t_sample time")
                    # get teleop command
                    sm_state = sm.get_motion_state()
                    # print(f"sm_state: {sm_state}")
                    
                    # print(f"sm_state{sm_state}")
                    dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                    # if np.any(dpos != 0):
                    #     print(f"dpos: {dpos}")
                    
                    drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)
                    drot_xyz[:] = 0
                    
                    # # commented out since no rotation is needed
                    # if not sm.is_button_pressed(0):
                    #     # translation mode
                    #     drot_xyz[:] = 0
                    # else:
                    #     dpos[:] = 0
                    # if not sm.is_button_pressed(1):
                    #     # 2D translation mode
                    #     dpos[2] = 0    

                    # print("Before euler")
                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    

                    # old_pose = target_pose.copy()
                    # print(f"old target pose: {old_pose}")
                    target_pose[:3] += dpos
                    target_pose[3:] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:])).as_rotvec()
                    # int(f"target_pose: {target_pose}")
                    

                    # execute teleop command
                    env.exec_actions(
                        actions=[target_pose], 
                        timestamps=[t_command_target-time.monotonic()+time.time()]
                        )
                    precise_wait(t_cycle_end)
                    iter_idx += 1



                    if exit_while_loop:
                        pause_enabled = False
                        print("Exiting human control loop")
                        terminate = False
                        break
                    
                    
               
                # ========== policy control loop ==============
                try:
                    # start episode
                    print("Policy in control!")
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        print('get_obs')
                        obs = env.get_obs()

                        print ("observation", obs)
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # print("result", result)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
                            print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions
                        if delta_action:
                            assert len(action) == 1
                            if perv_target_pose is None:
                                perv_target_pose = obs['robot_eef_pose'][-1]
                            this_target_pose = perv_target_pose.copy()
                            this_target_pose[[0,1]] += action[-1]
                            perv_target_pose = this_target_pose
                            this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        else:
                            this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float64)
                            this_target_poses[:] = target_pose
                            this_target_poses[:,[0,1]] = action
                            print("initially predicted target poses", this_target_poses)

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # # clip actions

                        # unclipped_target_poses = this_target_poses.copy()
                        # this_target_poses[:,:2] = np.clip(
                        #     this_target_poses[:,:2], [-0.8, -0.63], [-0.25, 0])
                        
                        # if np.any(this_target_poses != unclipped_target_poses):
                        #     print("Clipped target poses:")
                        #     print("Unclipped:", unclipped_target_poses)
                        #     print("Clipped:", this_target_poses)

                        print("sent target poses", this_target_poses)

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # if pause_enabled:
                        #     pause_start_time = time.monotonic()
                        #     print("Paused. Press 'r' to resume...")
                        #     exit_pause_loop = False
                        #     while True:
                        #         press_events = key_counter.get_press_events()
                        #         # print(f"Press events during pause: {press_events}")
                        #         for key_stroke in press_events:
                        #             if key_stroke == KeyCode(char='r'):
                        #                 # paused_duration = time.monotonic() - pause_start_time
                        #                 # print("paused_duration:", paused_duration)
                        #                 # total_paused_time = total_paused_time + paused_duration
                        #                 # print("total_paused_time:", total_paused_time)
                        #                 print(f"Resumed after seconds pause.")
                        #                 exit_pause_loop = True
                        #                 break
                                    
                                    
                        #         if exit_pause_loop:
                        #             print("Exited pause loop.")
                        #             break
                                

                        # # visualize
                        # episode_id = env.replay_buffer.n_episodes
                        # vis_img = obs[f'camera_{vis_camera_idx}'][-1]
                        # text = 'Episode: {}, Time: {:.1f}'.format(
                        #     episode_id, time.monotonic() - t_start
                        # )
                        # cv2.putText(
                        #     vis_img,
                        #     text,
                        #     (10,20),
                        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        #     fontScale=0.5,
                        #     thickness=1,
                        #     color=(255,255,255)
                        # )
                        # cv2.imshow('default', vis_img[...,::-1])

                        # print("waiting for stop press event")

                        terminate = False
                        press_events = key_counter.get_press_events()
                        exit_stop_while_loop = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print("pess events in stop loop:", press_events)
                                print('Stopping episode...')
                                env.end_episode()
                                terminate = True
                                break
                            
                            
                                
                        

                       
                        
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        term_pose = np.array([ 3.40948500e-01,  2.17721816e-01,  4.59076878e-02,  2.22014183e+00, -2.22184883e+00, -4.07186655e-04])
                        curr_pose = obs['robot_eef_pose'][-1]
                        dist = np.linalg.norm((curr_pose - term_pose)[:2], axis=-1)
                        if dist < 0.03:
                            # in termination area
                            curr_timestamp = obs['timestamp'][-1]
                            if term_area_start_timestamp > curr_timestamp:
                                term_area_start_timestamp = curr_timestamp
                            else:
                                term_area_time = curr_timestamp - term_area_start_timestamp
                                if term_area_time > 0.5:
                                    terminate = True
                                    print('Terminated by the policy!')
                        else:
                            # out of the area
                            term_area_start_timestamp = float('inf')

                        if terminate:
                            print("episode terminated")
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
