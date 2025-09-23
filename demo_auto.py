"""
ROBOT SPEED: 50% to make it smoother

Usage:
- go into directory /workspace/diffusion_policy
- then execute `python3 demo_real_robot.py -o /workspace/data/insertion_data_v2 --robot_ip 134.28.40.74`
- if you want to initalize the joints for the insertion task: 
`python3 demo_real_robot.py -o /workspace/data/insertion_data_v2 --robot_ip 134.28.40.74 --init_joints`  (please note that init_joint is a flag!)

Robot movement:
- left joystick controls robot EEF position in xy plane.
- right bumper/trigger controls robot EEF position in z axis.
- right joystick controls robot EEF rotation around z axis.

"""

# %%


# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env_gripper import RealEnv
from diffusion_policy.real_world.xbox_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode)
from diffusion_policy.common.transformation_related_functions import rotate_around_local_z
import traceback

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="Robot IP address e.g. 134.28.40.74")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
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
                    shm_manager=shm_manager
                ) as env:
            
            print('Environment initialized.')

            cv2.setNumThreads(1)
            
            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1.0)

            state = env.get_robot_state()
            print("Program start robot state:", state)

            target_pose = state['TargetTCPPose']
            action_array = np.zeros(7)
            action_array[:6] = target_pose[:]
            action_array[6] = 0
            inital_yaw_angle = target_pose[3:]
            print("Program start target pose:", target_pose)

            print("----------------------------\nPress 'c' to start recording, 's' to stop recording, 'q' to exit.")

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            auto_record_running = False
            auto_record_request_stop = False
            auto_record_stage = 0
            auto_record_target = None
            timestamp = None
            intial_run = True

            NEW_SAMPLE_TOL = 0.01       # 10mm
            DEVIATION_VAL = 0.005       # 5mm
            POS_REACHED_TOL = 0.001     # 1mm 
            ROT_REACHED_TOL = 0.001   
            RE_CENTER_DIST = 0.007 / 2  # 3.5mm

            while not stop:
                try:

                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # handle key presses
                    press_events = key_counter.get_press_events()
                    # record control via keyboard
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='/') and not is_recording:
                            # Exit program
                            stop = True
                        elif key_stroke == Key.f2 and not is_recording:
                            # Start recording
                            # if text_conditioning and not env.is_goal_text_valid():
                            #     print("Current text goal is not valid! Please check the format.")
                            #     continue
                            env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                            key_counter.clear()
                            is_recording = True
                            print('Recording!')
                        elif key_stroke == Key.f3 and not is_recording:
                            # Stop recording
                            env.end_episode()
                            key_counter.clear()
                            is_recording = False
                            print('Stopped.')
                        elif key_stroke == KeyCode(char='-') and not is_recording:
                            # Delete the most recent recorded episode
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                                is_recording = False
                            # delete

                        # auto mode
                        elif key_stroke == Key.f5 and not auto_record_running:
                            # Start auto-recording
                            auto_record_running = True
                            auto_record_request_stop = False
                            print('Auto-recording started!')
                        elif key_stroke == Key.f6 and auto_record_running: # south, A button
                            # Stop auto-recording
                            auto_record_request_stop = True
                            print('Auto-recording request stop!')

                    stage = key_counter[Key.space]
                    


                    precise_wait(t_sample)

                    if auto_record_running:

                        # get current robot eef pose (loop)
                        current_pose = obs['robot_eef_pose'][-1].copy()

                        if auto_record_stage == 0:
                            if intial_run:
                                target_pose[2] = 0.08 
                                intial_run = False
                                action_array[6] = 1

                            print(f"Stage {auto_record_stage}: Preparing next episode")
                            # robot has object inside gripper
                            auto_record_stage = 1
                            print(f"Entering stage {auto_record_stage}: object in gripper. Target pose {target_pose}")

                        if auto_record_stage == 1:
                                while True:
                                    auto_record_target = current_pose.copy()
                                    auto_record_target[:2] = sample_random_target()
                                    # only advance stage if the target is valid
                                    if (np.abs(auto_record_target[0] - current_pose[0]) >= NEW_SAMPLE_TOL and
                                        np.abs(auto_record_target[1] - current_pose[1]) >= NEW_SAMPLE_TOL):

                                        auto_record_stage = 2
                                        target_pose[:2] = auto_record_target[:2]
                                        target_pose[2] = 0.08
                                        print(f"Stage {auto_record_stage-1}: object in gripper, sampled new target {auto_record_target}")
                                        print(f"Entering stage {auto_record_stage}: Move to new target position")
                                        break
                                    else:
                                        print("Sampled target too close, will resample in next cycle")





                        if auto_record_stage == 2: # sample random rotation
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL: # sampled  position reached

                                # sample random grasping angle
                                auto_record_target = current_pose.copy()
                                object_rotation = np.random.uniform(-45, 45)
                                object_rotated_pose = current_pose.copy()

                                object_rotated_pose[3:] = st.Rotation.from_rotvec(object_rotated_pose[3:]).as_euler('xyz') # convert to euler
                                object_rotated_pose = rotate_around_local_z(object_rotated_pose, object_rotation) # apply transformation
                                object_rotated_pose[3:] = st.Rotation.from_euler('xyz',object_rotated_pose[3:]).as_rotvec() #convert back to rotvec


                                print(f"Stage {auto_record_stage}: object in gripper, sampled new object rotation {auto_record_target}")

                                # move towards new (sampled) object rotation
                                auto_record_stage = 3
                                target_pose = current_pose.copy()
                                target_pose[3:] = object_rotated_pose[3:] 

                                print(f"Entering stage {auto_record_stage}: Move to new target position")
                            else:
                                pass



                        if auto_record_stage == 3: # lifting
                            if np.linalg.norm(current_pose[3:] - target_pose[3:6]) < POS_REACHED_TOL: # rotation done
                                
                                action_array[6] = 0 # open gripper when roation is done
                                print(f"Stage {auto_record_stage}: Updated auto record target {auto_record_target}")

                                # target reached, move to next stage
                                auto_record_stage = 4
                                 
                                target_pose[2] = 0.12  # lift the robot eef 10 mm
                                print(f"Entering stage {auto_record_stage}: Will now lift the robot eef with target pose {target_pose}")
                                
                            else:
                                pass # wait until robot eef is re-centered to insertion frame

                        if auto_record_stage == 4: # move to starting position
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL: # target reached
                                print(f"Stage {auto_record_stage}: Lifted robot eef")

                                # now move to random new position
                                auto_record_stage = 5
                                target_pose = current_pose.copy()
                                target_pose[0] += np.random.uniform(-0.1, 0.1)
                                target_pose[1] += np.random.uniform(-0.1, 0.1)
                                print(f"Entering stage {auto_record_stage}: Will now go to random starting position for next episode: {target_pose}")

                            else:
                                pass # wait until robot eef is lifted

                        
                        if auto_record_stage == 5: # rotate to initial yaw
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL: # target reached
                                print(f"Stage {auto_record_stage}: Random starting position reached {target_pose}., reset yaw angle")
                                print(f",returned to initial yaw angle at starting position {auto_record_target}")

                                # move towards intial yaw angle
                                auto_record_stage = 6
                                target_pose = current_pose.copy()
                                target_pose[3:] = inital_yaw_angle 

                                print(f"Entering stage {auto_record_stage}: will start episode")
                            else:
                                pass



                        if auto_record_stage == 6: # move to initial devaited position
                            if np.linalg.norm(current_pose[3:] - target_pose[3:6]) < POS_REACHED_TOL: # target reached
                                # target reached, start recording
                                print(f"Stage {auto_record_stage}: Random starting position reached {target_pose}.")
                                print(f"Stage {auto_record_stage}: Start recording!")

                                env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                                key_counter.clear()
                                is_recording = True

                                # add some deviation to the target pose (only x and y)
                                auto_record_stage = 7
                                target_pose[:2] = auto_record_target[:2]  # only take initial 3 values of saveed target
                                target_pose[0] += np.random.uniform(-DEVIATION_VAL, DEVIATION_VAL)
                                target_pose[1] += np.random.uniform(-DEVIATION_VAL, DEVIATION_VAL)
                                print(f"Entering stage {auto_record_stage}: will go to approach point.")

                            else:
                                pass # wait until robot eef is at new target position from frame

                        if auto_record_stage == 7:
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                print(f"Stage {auto_record_stage}: Target position reached {target_pose}")

                                # move to "perfect" target pose
                                auto_record_stage = 8
                                target_pose[:2] = auto_record_target[:2]  # only take initial 3 values of saveed target
                                print(f"Entering stage {auto_record_stage}: Going to perfect approach point")

                            else:
                                pass # wait until robot eef is at new target position from frame

                        
                        if auto_record_stage == 8: # rotate back to object rotation
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL: # target reached
                                print(f"Stage {auto_record_stage}: engage point reached {target_pose}., rotating gripper")

                                
                                target_pose[3:] = object_rotated_pose[3:]

                                print(f"Stage {auto_record_stage}: rotated grippeer to align with object {auto_record_target}")

                                auto_record_stage = 9

                                print(f"Entering stage {auto_record_stage}: will start downward motion")
                            else:
                                pass

                        if auto_record_stage == 9:
                            if np.linalg.norm(current_pose[3:] - target_pose[3:6]) < POS_REACHED_TOL:
                                print(f"Stage {auto_record_stage}: Perfect target position reached {target_pose}.")
                                
                                auto_record_stage = 10
                                target_pose[2] = auto_record_target[2]  # engage with insertion frame
                                print(f"Entering stage {auto_record_stage}: Engage")
                            
                            else:
                                pass # wait until robot eef is at new target position from frame

                        if auto_record_stage == 10:
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                action_array[6] = 1
                                print(f"Stage {auto_record_stage}: Engaged with insertion frame, will now wait for 3seconds")
                                timestamp = time.monotonic()
                                auto_record_stage = 11

                        if auto_record_stage == 11:
                            if time.monotonic() - timestamp >= 5.0:
                                print(f"Stage {auto_record_stage}: Waited for 5 seconds, will now move to next episode")

                                # Stop recording
                                env.end_episode()
                                key_counter.clear()
                                is_recording = False
                                print('Stopped.')

                                # retrun to initial yaw angle
                                target_pose[3:] = inital_yaw_angle
                                auto_record_stage = 0

                                if auto_record_request_stop:
                                    print('Auto-recording stopped!')
                                    auto_record_running = False
                                    auto_record_request_stop = False

                            else:
                                pass # wait until 3 seconds are over
                        
                        drot = st.Rotation.from_euler('xyz', np.zeros(3, dtype=np.float32))

                   


                    # print("target_pose", target_pose)

                    # fixed values for the insertion task (fixate rotation to initial pose angles)
                    # fixed_val_rot_x = -2.90612772       # for the demo "insertion_locked_rot_euler" these values remain fixed such that the ee is aligned with table plane
                    # fixed_val_rot_y = 1.19331937
                    # fixed_val_rot_z = 1.47788891e-04
                    # target_pose[3] = fixed_val_rot_x
                    # target_pose[4] = fixed_val_rot_y
                    # target_pose[5] = fixed_val_rot_z

                    # execute teleop command
                    # add gripper to target pose 
                    
                    # print("sent target pose", target_pose)
                    action_array[:6] = target_pose
                    env.exec_actions( 
                        actions=[action_array], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        stages=[stage])
                    precise_wait(t_cycle_end)
                    iter_idx += 1

                except Exception as e:
                    print(f'EXCEPTION in {__file__}: {e}')
                    traceback.print_exc()


def sample_random_target(limits_x=[-0.75, -0.55], limits_y=[-0.35, -0.15], limits_z=[0.08, 0.15]):
    """
    Sample a random target pose (x, y, z) within given limits.
    """
    x = np.random.uniform(limits_x[0], limits_x[1])
    y = np.random.uniform(limits_y[0], limits_y[1])
    z = np.random.uniform(limits_z[0], limits_z[1])
    return np.array([x, y], dtype=np.float32)


# %%
if __name__ == '__main__':
    main()
