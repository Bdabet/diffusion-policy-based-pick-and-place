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
@click.option('--text_conditioning', '-tcond', default = False, type= bool, help="policy conditioning using text (True/False)")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency, text_conditioning):
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
                    text_conditioned = text_conditioning
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
            inital_yaw_angle = target_pose
            print("Program start target pose:", target_pose)

            print("----------------------------\nPress 'c' to start recording, 's' to stop recording, 'q' to exit.")

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            FIXED_STARTING_POSITION = [-5.67546541e-01,  9.37104784e-02, 9.49910267e-02, 2.87012257e+00 ,-1.19877504e+00, -2.55008721e-03]
            POS_2 = [-5.08497330e-01, -2.54640933e-01, 7.50035058e-02,  2.87002445e+00, -1.19890213e+0, -2.42054737e-03]

            

            auto_record_running = False
            auto_record_request_stop = False
            auto_record_stage = 0
            current_object_starting_pos = None
            timestamp = None
            initial_run = True
            gripper_state = 0 # start gripper in open state
            object_is_cylinder = False
            initial_starting_position = None
            final_placing_position = POS_2
            objects = None

            NEW_SAMPLE_TOL = 0.07       # 7 cm
            DEVIATION_VAL = 0.005       # 5 mm
            FINAL_POS_VARIATION = 0.03
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
                    obs = env.get_obs(text_conditioned = text_conditioning)

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
                        elif key_stroke == Key.f3 and not auto_record_running:
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
                       

                        # === INITIALIZE MULTI-OBJECT EPISODE ===
                        if objects is None:
                            NUM_OBJECTS = 3  # number of objects per full episode
                            FINAL_POSITIONS = [final_pos1, final_pos2, final_pos3]  # define 3 distinct goal poses

                            objects = []
                            for i in range(NUM_OBJECTS):
                                obj = {
                                    'id': i,
                                    'auto_record_stage': 0,
                                    'random_start_pose': None,  # random initial position
                                    'random_rotated_start_pose': None,
                                    'final_pose': FINAL_POSITIONS[i].copy(),
                                    'done': False
                                }
                                objects.append(obj)

                            current_obj_idx = 0
                            episode_started = False
                            print(f"Initialized {NUM_OBJECTS} objects for continuous episode.")

                        # === PROCESS CURRENT OBJECT ===
                        obj = objects[current_obj_idx]
                        auto_record_stage = obj['auto_record_stage']

                        current_pose = obs['robot_eef_pose'][-1].copy()
                        
                        if initial_run:
                            timestamp = time.monotonic()

                        # === START EPISODE ON FIRST PICK ===
                        if not episode_started and auto_record_stage >= 6 and current_obj_idx < len(objects) - 1:
                            print("Starting full multi-object recording.")
                            if text_conditioning and env.is_goal_text_valid():
                                env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                                key_counter.clear()
                                is_recording = True
                                episode_started = True
                            else:
                                print("Invalid goal text. Cannot start episode.")
                                auto_record_running = False
                                return

                        if auto_record_stage == -1: #move to pick object for episode preparation
                            if np.linalg.norm(current_pose - target_pose) < POS_REACHED_TOL:
                                if initial_run:
                                    target_pose[0:2] = initial_starting_position[0:2]
                                    auto_record_stage = 0
                                    timestamp = time.monotonic()
                                else:
                                    target_pose[:2] = obj['final_pose'][:2]  # move to x,y of final placing position
                                    target_pose[3:] = obj['final_pose'][3:] # handle rotation
                                    timestamp = time.monotonic() + 5 # avoid waiting when not initial run

                        if auto_record_stage == 0 and time.monotonic() - timestamp >= 5:  # move downwards to object and close gripper
                            
                            
                            target_pose[2] = 0.075
                            initial_starting_position = current_pose.copy()
                            # intial_run = False
                            print(f"Stage {auto_record_stage}: Preparing next episode")
                            auto_record_stage = 0.25
                            print(f"Entering stage {auto_record_stage}: object in gripper. Target pose {target_pose}")

                        if auto_record_stage == 0.25:  # close gripper
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                action_array[6] = 1
                                auto_record_stage = 0.5
                                timestamp = time.monotonic()

                        if auto_record_stage == 0.5 and time.monotonic() - timestamp >= 0.5:
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                target_pose[2] = 0.09
                                auto_record_stage = 1

                        if auto_record_stage == 1: # sample new object random position and move to it
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                while True:
                                    current_object_starting_pos = current_pose.copy()
                                    current_object_starting_pos[:2] = sample_random_target()
                                    if (np.abs(current_object_starting_pos[0] - current_pose[0]) >= NEW_SAMPLE_TOL and
                                        np.abs(current_object_starting_pos[1] - current_pose[1]) >= NEW_SAMPLE_TOL):
                                        auto_record_stage = 2
                                        target_pose[:2] = current_object_starting_pos[:2]
                                        obj['random_start_pose'] = current_object_starting_pos.copy()  # assign random_start_pose here
                                        print(f"Stage {auto_record_stage-1}: sampled new target {current_object_starting_pos}")
                                        break
                                    else:
                                        print("Sampled target too close, will resample.")
                        if auto_record_stage == 2: # smaple random object angle and rotate
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                current_object_starting_pos = current_pose.copy()
                                current_object_starting_pos[2] = 0.075
                                if not object_is_cylinder:
                                    object_rotation = np.random.uniform(-45, 45)
                                    object_rotated_pose = current_pose.copy()
                                    object_rotated_pose[3:] = st.Rotation.from_rotvec(object_rotated_pose[3:]).as_euler('xyz')
                                    object_rotated_pose = rotate_around_local_z(object_rotated_pose, object_rotation)
                                    object_rotated_pose[3:] = st.Rotation.from_euler('xyz', object_rotated_pose[3:]).as_rotvec()
                                    obj['random_rotated_start_pose'] = object_rotated_pose.copy()
                                else:
                                    # For cylinder, set random_rotated_start_pose to current_pose to avoid KeyError
                                    obj['random_rotated_start_pose'] = current_pose.copy()
                                print(f"Stage {auto_record_stage}: sampled rotation {current_object_starting_pos}")
                                auto_record_stage = 2.5
                                target_pose = current_pose.copy()
                                if not object_is_cylinder:
                                    target_pose[3:] = object_rotated_pose[3:]
                                print(f"Entering stage {auto_record_stage}: moving to starting position.")
                                print(f"Entering stage {auto_record_stage}: moving to starting position.")

                        if auto_record_stage == 2.5: # move downwards place object at random starting pose
                            if np.linalg.norm(current_pose - target_pose) < POS_REACHED_TOL:
                                target_pose[2] = 0.075
                                auto_record_stage = 3

                        if auto_record_stage == 3: # open gripper at object starting random pose and lift
                            if np.linalg.norm(current_pose - target_pose) < POS_REACHED_TOL:
                                action_array[6] = 0
                                print(f"Stage {auto_record_stage}: Updated auto record target {current_object_starting_pos}")
                                if current_obj_idx < len(objects) - 1:
                                    auto_record_stage = -1

                                    obj['start_pose'] = current_pose
                                    # Check if more objects remain
                                    current_obj_idx += 1

                                else:
                                    obj = objects[0] # loop back to first object for next steps
                                    auto_record_stage = 4
                                target_pose[2] = np.random.uniform(0.12, 0.15)
                                print(f"Entering stage {auto_record_stage+1}: lifting with target {target_pose}")



                        # === Episode is prepared ===


                        if auto_record_stage == 4: # move to set starting positon
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                print(f"Stage {auto_record_stage}: Lifted robot eef")
                                auto_record_stage = 5
                                target_pose = current_pose.copy()
                                target_pose[:2] = FIXED_STARTING_POSITION[:2]
                                target_pose[0] += np.random.uniform(-DEVIATION_VAL, DEVIATION_VAL)
                                target_pose[1] += np.random.uniform(-DEVIATION_VAL, DEVIATION_VAL)
                                print(f"Entering stage {auto_record_stage}: going to random start position {target_pose}")

                        if auto_record_stage == 5 and not object_is_cylinder: # rotate to deviated starting angle
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                print(f"Stage {auto_record_stage}: Random start reached.")
                                auto_record_stage = 6
                                target_pose = current_pose.copy()
                                varied_initial_yaw_angle = inital_yaw_angle.copy()
                                varied_initial_yaw_angle[3:] = st.Rotation.from_rotvec(varied_initial_yaw_angle[3:]).as_euler('xyz')
                                varied_initial_yaw_angle = rotate_around_local_z(varied_initial_yaw_angle, np.random.uniform(-2, 2))
                                varied_initial_yaw_angle[3:] = st.Rotation.from_euler('xyz', varied_initial_yaw_angle[3:]).as_rotvec()
                                target_pose[3:] = varied_initial_yaw_angle[3:]
                                print(f"Entering stage {auto_record_stage}: will start pick sequence.")
                        elif auto_record_stage == 5 and object_is_cylinder:
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                auto_record_stage = 6

                        if auto_record_stage == 6: # move to deviated approach point above object (skipped)
                            if np.linalg.norm(current_pose[3:] - target_pose[3:6]) < POS_REACHED_TOL: # target reached
                                # target reached, start recording
                                print(f"Stage {auto_record_stage}: Random starting position reached {target_pose}.")
                                print(f"Stage {auto_record_stage}: Start recording!")#
                                if text_conditioning and env.is_goal_text_valid():
                                    env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                                    key_counter.clear()
                                    is_recording = True
                                    # add some deviation to the target pose (only x and y)
                                    auto_record_stage = 7
                                    print(f"target pose {target_pose}, auto_record_target {current_object_starting_pos}")
                                    target_pose[:2] = obj['random_start_pose'][:2]  # only take initial 3 values of saved target
                                    target_pose[0] += np.random.uniform(-0, 0) 
                                    target_pose[1] += np.random.uniform(-0, 0)
                                    print(f"Entering stage {auto_record_stage}: will go to approach point.")
                                else:
                                    print("Current text goal is not valid! Please check the format.")
                                    auto_record_running = False

                            else:
                                pass # wait until robot eef is at new target position from frame

                        if auto_record_stage == 7: # move to final lifted position before rotation
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                print(f"Stage {auto_record_stage}: Target position reached {target_pose}")

                                # move to "perfect" target pose
                                auto_record_stage = 8
                                target_pose[:2] = obj['random_start_pose'][:2]  # only take initial 2 values of saved target
                                print(f"Entering stage {auto_record_stage}: Going to perfect approach point")

                            else:
                                pass # wait until robot eef is at new target position from frame

                        
                        if auto_record_stage == 8 and not object_is_cylinder: # rotate back to object rotation
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL: # target reached
                                
                                print(f"Stage {auto_record_stage}: engage point reached {target_pose}., rotating gripper")
                                target_pose[3:] = obj['random_rotated_start_pose'][3:]
                                print(f"Stage {auto_record_stage}: rotated grippeer to align with object {current_object_starting_pos}")
                                auto_record_stage = 9
                                print(f"Entering stage {auto_record_stage}: will start downward motion")
                            else:
                                pass
                        elif auto_record_stage == 8 and object_is_cylinder:
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL: # target reached
                                auto_record_stage = 9

                        if auto_record_stage == 9: # move downwards into object 
                            if np.linalg.norm(current_pose[3:] - target_pose[3:6]) < POS_REACHED_TOL:
                                print(f"Stage {auto_record_stage}: Perfect target position reached {target_pose}.")
                                
                                auto_record_stage = 10
                                target_pose[2] = obj['random_start_pose'][2]  # move downwards into object
                                print(f"Entering stage {auto_record_stage}: Engage")
                                timestamp = time.monotonic()
                            
                            else:
                                pass # wait until robot eef is at new target position from frame

                        if auto_record_stage == 10 and time.monotonic() - timestamp >= 1: # wait then close gripper
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                action_array[6] = 1
                                print(f"Stage {auto_record_stage}: Engaged with insertion frame, will now wait for 3seconds")
                                timestamp = time.monotonic()
                                auto_record_stage = 11

                        
                        if auto_record_stage == 11 and time.monotonic() - timestamp >= 1: # wait then lift to horizontal moving height 
                            if np.linalg.norm(current_pose - target_pose) < POS_REACHED_TOL: # rotation done
                                print(f"Stage {auto_record_stage}: lifting to horizontal moving height")
                                # target reached, move to next stage
                                auto_record_stage = 12
                                 
                                target_pose[2] = 0.15 # lift the robot eef 15 mm
                                print(f"Entering stage {auto_record_stage}: Will now lift the robot eef with target pose {target_pose}")
                                
                            else:
                                pass # wait until robot eef is re-centered to insertion frame
                        
                        if auto_record_stage == 12: # move horizontally to final placing position
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                print(f"stage {auto_record_stage}: moving to final position.")
                                target_pose[:2] = obj['final_placing_position'][:2] # only need x,y coordinates
                                auto_record_stage = 13


                        if auto_record_stage == 13: # rotate to final orientation
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                print(f"stage {auto_record_stage}: rotating to final orientation.")
                                target_pose[3:] = obj['final_placing_position'][3:]
                                auto_record_stage = 14

                        if auto_record_stage == 14: # move downwards
                             if np.linalg.norm(current_pose[3:] - target_pose[3:]) < POS_REACHED_TOL:
                                  print(f"stage {auto_record_stage}: moving downwards to place object.")
                                  target_pose[2] = 0.075
                                  auto_record_stage = 15
                    

                        if auto_record_stage == 15: # open gripper 
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                action_array[6] = 0
                                auto_record_stage = 16
                                timestamp = time.monotonic()


                        # === FINAL STAGE MODIFICATION ===
                        if auto_record_stage == 16:
                            if time.monotonic() - timestamp >= 4.0:
                                print(f"Stage {auto_record_stage}: Finished object {obj['id']}")
                                action_array[6] = 0
                                is_recording = False
                                obj['done'] = True
                                obj['auto_record_stage'] = 0  # reset

                                # Check if more objects remain
                                if current_obj_idx < len(objects) - 1:
                                    current_obj_idx += 1
                                    print(f"Moving to next object {current_obj_idx}")
                                else:
                                    print("All objects placed — ending episode.")
                                    env.end_episode()
                                    auto_record_running = False
                                    episode_started = False
                                    objects = None

                        # Update stage state for current object
                        obj['auto_record_stage'] = auto_record_stage
                        action_array[:6] = target_pose

                    else:
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
                        if np.any(dpos != 0):
                            print(f"dpos: {dpos}")

                        
                        
                        target_pose[:3] += dpos

                        action_array[:6] = target_pose
                        action_array[6] = float(gripper_state == True)
                    
                    # print("sent target pose", target_pose)
                    
                    env.exec_actions( 
                        actions=[action_array], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        stages=[stage])
                    precise_wait(t_cycle_end)
                    iter_idx += 1

                except Exception as e:
                    print(f'EXCEPTION in {__file__}: {e}')
                    traceback.print_exc()


def sample_random_target(limits_x=[-0.85, -0.32], limits_y=[-0.45, -0.08], limits_z=[0.08, 0.15]):
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
