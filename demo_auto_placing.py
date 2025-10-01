"""
ROBOT SPEED: 50% to make it smoother

Usage:
- go into directory /workspace/diffusion_policy
- then execute `python3 demo_real_robot.py -o /workspace/data/insertion_data_v2 --robot_ip 134.28.40.74`
- if you want to initalize the joints for the insertion task: 
`python3 demo_auto_placing.py -o /workspace/data/insertion_data_v2 --robot_ip 134.28.40.74 --init_joints`  (please note that init_joint is a flag!)

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
            # action_array[:6] = target_pose[:]
            action_array[6] = 0
            inital_yaw_angle = target_pose
            print("Program start target pose:", target_pose)

            print("----------------------------\nPress 'c' to start recording, 's' to stop recording, 'q' to exit.")

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            POS_1= [-7.10899248e-01, -2.56950736e-01,  6.30217522e-02,  2.87004895e+00, -1.19886725e+00, -2.42536075e-03]
            POS_2 = [-5.08497330e-01, -2.54640933e-01, 7.50035058e-02,  2.87002445e+00, -1.19890213e+0, -2.42054737e-03]
            POS_3 = [-3.24455625e-01, -2.48806794e-01,  7.50026995e-02,  2.87006066e+00, -1.19883497e+00, -2.45482265e-03]

            auto_record_running = False
            auto_record_request_stop = False
            auto_record_stage = 0
            deviated_initial_starting_positon = None
            timestamp = None
            intial_run = True
            gripper_state = 0 # start gripper in open state
            object_is_cylinder = True
            inital_starting_pose = None
            final_placing_position = POS_1

            
            NEW_SAMPLE_TOL = 0.01       # 10mm
            DEVIATION_VAL = 0.005       # 5mm
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

                        # get current robot eef pose (loop)
                        current_pose = obs['robot_eef_pose'][-1].copy()
                        current_gripper_state = obs['gripper_state'][0]

                        if auto_record_stage == 0: # move downwards to object (only in first run)
                            if intial_run:
                                target_pose[2] = 0.075 
                                intial_run = False
                            
                                    
                            print(f"Stage {auto_record_stage}: Preparing next episode")
                            # robot has object inside gripper
                            auto_record_stage = 0.25
                            print(f"Entering stage {auto_record_stage}: object in gripper. Target pose {target_pose}")
                        
                        if auto_record_stage == 0.25: # close gripper
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                action_array[6] = 1
                                auto_record_stage = 0.5
                                timestamp = time.monotonic()

                        if auto_record_stage == 0.5 and time.monotonic() - timestamp >= 0.5: # wait for a second to close gripper before lifting for movement
                            if np.linalg.norm(current_pose - target_pose) < POS_REACHED_TOL and current_gripper_state == 1:
                                target_pose[2] = 0.09
                                auto_record_stage = 1



                        if auto_record_stage == 1:  # sample new starting position and carry it out
                                if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                    if np.any(inital_starting_pose) == None:
                                        inital_starting_pose = current_pose.copy()
                                    while True:
                                        deviated_initial_starting_positon = current_pose.copy()
                                        deviated_initial_starting_positon[0] = inital_starting_pose[0] + np.random.uniform(-FINAL_POS_VARIATION, FINAL_POS_VARIATION)
                                        deviated_initial_starting_positon[1] = inital_starting_pose[1] + np.random.uniform(-FINAL_POS_VARIATION, FINAL_POS_VARIATION)
                                        # only advance stage if the target is valid
                                        if (np.abs(deviated_initial_starting_positon[0] - current_pose[0]) >= NEW_SAMPLE_TOL and
                                            np.abs(deviated_initial_starting_positon[1] - current_pose[1]) >= NEW_SAMPLE_TOL): # TOL = 0.01

                                            auto_record_stage = 2
                                            target_pose[:2] = deviated_initial_starting_positon[:2]
                                            # target_pose[2] = 0.08 # no need to move back downwards before episode starts
                                            print(f"Stage {auto_record_stage-1}: object in gripper, sampled new target {deviated_initial_starting_positon}")
                                            print(f"Entering stage {auto_record_stage}: sample new rotation")
                                            break
                                        else:
                                            print("Sampled target too close, will resample in next cycle")


                        if auto_record_stage == 2: # sample and carry out random rotation
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL: # sampled  position reached

                                # sample random grasping angle
                                deviated_initial_starting_positon = current_pose.copy()
                                deviated_initial_starting_positon[2] = 0.075 # target should be below when object touches base
                                if object_is_cylinder:
                                    object_rotation = np.random.uniform(-2, 2)
                                   
                                else:
                                    object_rotation = np.random.uniform(-80, 80)
                                    
                                object_rotated_pose = current_pose.copy()
                                object_rotated_pose[3:] = st.Rotation.from_rotvec(object_rotated_pose[3:]).as_euler('xyz') # convert to euler
                                object_rotated_pose = rotate_around_local_z(object_rotated_pose, object_rotation) # apply transformation
                                object_rotated_pose[3:] = st.Rotation.from_euler('xyz',object_rotated_pose[3:]).as_rotvec() #convert back to rotvec


                                print(f"Stage {auto_record_stage}: object in gripper, sampled new object rotation {deviated_initial_starting_positon}")

                                # carry out (sampled) object rotation
                                auto_record_stage = 2.5
                                target_pose = current_pose.copy()
                                target_pose[3:] = object_rotated_pose[3:] 

                                print(f"Entering stage {auto_record_stage}: move back downwards before starting episode")
                            else:
                                pass
                        
                        
                        if auto_record_stage == 2.5: # move back downwards before starting episode
                            if np.linalg.norm(current_pose - target_pose) < POS_REACHED_TOL: # sampled  position reached
                                target_pose[2] = 0.075
                                auto_record_stage = 3

                        

                        if auto_record_stage == 3: # lifting to horizontal moving height 
                            if np.linalg.norm(current_pose - target_pose) < POS_REACHED_TOL: # rotation done
                                if text_conditioning and env.is_goal_text_valid():
                                    # start episode
                                    env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                                    key_counter.clear()
                                    is_recording = True
                                else:
                                    print("Current text goal is not valid! Please check the format.")
                                    auto_record_running = False
                                print(f"Stage {auto_record_stage}: started episode")
                                # target reached, move to next stage
                                auto_record_stage = 4
                                 
                                target_pose[2] = 0.15 # lift the robot eef 10 mm
                                print(f"Entering stage {auto_record_stage}: Will now lift the robot eef with target pose {target_pose}")
                                
                            else:
                                pass # wait until robot eef is re-centered to insertion frame
                        
                        if auto_record_stage == 4: # move horizontally to final placing position
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                print(f"stage {auto_record_stage}: moving to final position.")
                                target_pose[:2] = final_placing_position[:2] # only need x,y coordinates
                                auto_record_stage = 5


                        if auto_record_stage == 5: # rotate to final orientation
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                print(f"stage {auto_record_stage}: rotating to final orientation.")
                                target_pose[3:] = final_placing_position[3:]
                                auto_record_stage = 6
                            

                        if auto_record_stage == 6: # move downwards 
                             if np.linalg.norm(current_pose[3:] - target_pose[3:]) < POS_REACHED_TOL:
                                  print(f"stage {auto_record_stage}: moving downwards to place object.")
                                  target_pose[2] = 0.075
                                  auto_record_stage = 7
                    

                        if auto_record_stage == 7: # open gripper 
                            if np.linalg.norm(current_pose[:3] - target_pose[:3]) < POS_REACHED_TOL:
                                action_array[6] = 0
                                auto_record_stage = 8
                                timestamp = time.monotonic()


                        if auto_record_stage == 8: # time delay
                            if time.monotonic() - timestamp >= 5.0:
                                print(f"Stage {auto_record_stage}: Waited for 5 seconds, will now move to next episode")

                                # Stop recording
                                env.end_episode()
                                is_recording = False
                                print('Stopped.')

                                # return to initial pose
                                action_array[6] = 1
                                auto_record_stage = 0

                                

                                if auto_record_request_stop:
                                    print('Auto-recording stopped!')
                                    auto_record_running = False
                                    auto_record_request_stop = False
                                    inital_starting_pose = None
                                    action_array[6] = 0

                            else:
                                pass # wait until 5 seconds are over
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
                    
                    # current_pose = obs['robot_eef_pose'][-1].copy()
                    # print("current pose:", current_pose)

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
