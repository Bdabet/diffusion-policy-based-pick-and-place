"""
Usa
(robodiff)$ python3 demo_real_robot_gripper.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env_gripper import RealEnv
# from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
# from diffusion_policy.real_world.spacemouse_shared_memory_modified import Spacemouse
from diffusion_policy.real_world.xbox_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode)
from diffusion_policy.common.transformation_related_functions import rotate_around_local_z


@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--text_conditioning', '-tcond', default = False, type= bool, help="policy conditioning using text (True/False)")
@click.option('--image_conditioning', '-icond', default = False, type= bool, help="policy conditioning using image (True/False)")
@click.option('--quaternion', '-q', default=False, help="Whether to use quaternion for rotation representation.")
@click.option('--balanced_configs', '-bc', default=False, help="Whether to use balanced configurations for text goals.")
@click.option('--no_of_rep', '-nr', default=1, help="Number of repetitions for each configuration.")


def main(output, robot_ip,  init_joints, frequency, command_latency, text_conditioning, image_conditioning, quaternion, balanced_configs, no_of_rep):
    dt = 1/frequency
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            Spacemouse(shm_manager=shm_manager) as sm, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                # recording resolution
                obs_image_resolution=(1280,720),
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
                image_conditioned=image_conditioning,   
                text_conditioned=text_conditioning,
                balanced_configs=balanced_configs,
                no_of_repetitions=no_of_rep,
            ) as env:

            cv2.setNumThreads(1)
            

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            
            
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            
            time.sleep(1.0)
            print('Ready!')
            state = env.get_robot_state()
            print(f'Robot state: {state}')

            # initialze target pose, action array and set gripper state to 0
            target_pose = state['TargetTCPPose']
            action_array = np.zeros(7)
            
            # start the grpper in open state
            gripper_state = 0

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                env.get_obs(text_conditioned = text_conditioning)

                # print("waiting for press events")

                # handle key presses
                press_events = key_counter.get_press_events()
                # print(f"press_events: {press_events}")
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='/'):
                        # Exit program
                        stop = True
                    elif key_stroke == Key.f2:
                        # Start recording
                        if text_conditioning and not env.is_goal_text_valid():
                            print("Current text goal is not valid! Please check the format.")
                            continue
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == Key.f3:
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == KeyCode(char='-'):
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                stage = key_counter[Key.space]
                # print(f"stage: {stage}")



                precise_wait(t_sample)
                
                # get teleop command
                sm_state = sm.get_motion_state()

                # print("sm state", sm_state)


                # handle gripper commands
                button_cicked = sm.get_button_state()
                if button_cicked[0]:
                    gripper_state = not gripper_state

                # handle rotation commands
                if button_cicked[1]:
                    # convert target pose to euler format
                    target_pose[3:] = st.Rotation.from_rotvec(target_pose[3:]).as_euler('xyz')
                    # rotate target pose around local z
                    target_pose = rotate_around_local_z(target_pose, 1)
                    # convert target pose back to rotvec
                    target_pose[3:]= st.Rotation.from_euler('xyz',target_pose[3:]).as_rotvec()


                elif button_cicked[2]:

                    # convert target pose to euler format
                    target_pose[3:] = st.Rotation.from_rotvec(target_pose[3:]).as_euler('xyz')
                    # rotate target pose around local z
                    target_pose = rotate_around_local_z(target_pose, -1)
                    # convert target pose back to rotvec
                    target_pose[3:]= st.Rotation.from_euler('xyz',target_pose[3:]).as_rotvec()
               
                # print("final target pose ", target_pose)


                dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                # if np.any(dpos != 0):
                #     print(f"dpos: {dpos}")

                
                
                target_pose[:3] += dpos

                action_array[:6] = target_pose
                action_array[6] = float(gripper_state == True)


                # print("action array", action_array)
               
                
                # execute teleop command
                
                env.exec_actions(
                    actions=[action_array], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage],
                    quaternions=quaternion)
                precise_wait(t_cycle_end)   
                iter_idx += 1

                # print(f"iter_idx: {iter_idx}, t_cycle_end: {t_cycle_end}, t_command_target: {t_command_target}")





if __name__ == '__main__':
    main()