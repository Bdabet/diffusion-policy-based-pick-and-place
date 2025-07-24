"""
Usa
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

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
from scipy.spatial.transform import Rotation as R

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
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

            # print("RealEnv initialized! 2")

            cv2.setNumThreads(1)
            

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            
            
            # realsense white balance
            env.realsense.set_white_balance(white_balance=5900)

            
            time.sleep(1.0)
            print('Ready!')
            state = env.get_robot_state()
            print(f'Robot state: {state}')
            target_pose = state['TargetTCPPose']
            action_array = np.zeros(13)
            # action_array = np.append(action_array, 0) # add extra values for gripper
            # print(f'Robot target pose + other values: {action_array}')

            # initialize gripper state to 0
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

                # inital gripper state always set rto 0


                robot_state = env.robot.get_all_state()


                # print ("shape", np.shape(robot_state["ActualQ"])) #(30,6)
                # print("current angles",robot_state["ActualQ"][-1, :])

                current_joint_angles = robot_state["ActualQ"][-1, :]



                # print("waiting for press events")


                # handle key presses
                press_events = key_counter.get_press_events()
                # print(f"press_events: {press_events}")
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='l'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='j'):
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='k'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
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

                print("sm state", sm_state)



                button_cicked = sm.get_button_state()
                if button_cicked[0]:
                    gripper_state = not gripper_state


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
                



                # print(f"sm_state: {sm_state}")
                

                dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                if np.any(dpos != 0):
                    print(f"dpos: {dpos}")

                
                
                target_pose[:3] += dpos

                
                
                
                
                
                print("final target psoe ", target_pose)



                # print("new j angles", new_joint_angles)
                # int(f"target_pose: {target_pose}")

                
                
                


                # if button_cicked[1]:
                #     # clockwise rotation
                #     new_joint_angles = current_joint_angles.copy()
                #     # target_pose[5] = target_pose[5] + np.deg2rad(1)
                #     target_pose = rotate_around_local_z(target_pose, 1)
                #     # print("rotated pose", rotate_around_local_z(target_pose, 5))
                #     # print("clockwise", new_joint_angles)
                # elif button_cicked[2]:
                #     # aniclockwise rotation
                #     new_joint_angles = current_joint_angles.copy()
                #     # new_joint_angles[5] = new_joint_angles[5] - np.deg2rad(1)
                #     # target_pose[5] = target_pose[5] - np.deg2rad(1)
                # else:
                #     new_joint_angles = current_joint_angles.copy()

                


                action_array[:6] = target_pose
                action_array[6] = float(gripper_state == True)
                
                # action_array[6:12] = new_joint_angles
                
                # action_array = np.concatenate((target_pose, int(gripper_state == 'true'), new_joint_angles), axis = 0)
                # action_array[6:] = new_joint_angles[:] 
                
                
             

                

                print("action array", action_array)
               
                
                # execute teleop command
                
                env.exec_actions(
                    actions=[action_array], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)    # print("rotated pose", rotate_around_local_z(target_pose, 5))
                    # print("clockwise", new_joint_angles)if first_rot_received:
                #     assert rtde_c.servoJ(target_joint_angles,
                #             vel, acc, # dummy, not used by ur5
                #             dt, 
                #             self.lookahead_time, 
                #             self.gain)
                iter_idx += 1
                # print(f"iter_idx: {iter_idx}, t_cycle_end: {t_cycle_end}, t_command_target: {t_command_target}")


def rotate_around_local_z(grasping_pose, rotation_angle, robot_angle_offset = 45):
    
    # Current rotation based on the grasping pose
    r_current = R.from_euler('xyz', grasping_pose[3:], degrees=False)

    
    r_local_z = R.from_euler('z', np.deg2rad(rotation_angle), degrees=False)
    
        

    # Combine the rotations
    r_new = r_current * r_local_z

    # debug 
    # print("r_new", r_new.as_euler('xyz', degrees=True))

    # Update the final grasping pose
    grasping_pose = np.concatenate((grasping_pose[:3], r_new.as_euler('xyz', degrees=False)), axis=None)

    return grasping_pose
# %%

def rotate_tcp_z_axis_rpy(pose_rpy: np.ndarray, angle_deg: float) -> np.ndarray:
    
    # Extract position and orientation
    position = pose_rpy[:3]
    rpy = pose_rpy[3:]

    # Convert RPY to rotation matrix
    rot = R.from_euler('xyz', rpy)
    rot_matrix = rot.as_matrix()

    # Define rotation around local Z-axis of TCP
    angle_rad = np.deg2rad(angle_deg)
    Rz = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                 0,                  1]
    ])

    # Apply rotation in TCP frame: R_new = R_current @ Rz
    new_rot_matrix = rot_matrix @ Rz
    new_rpy = R.from_matrix(new_rot_matrix).as_euler('xyz')

    # Combine new orientation with original position
    return np.concatenate([position, new_rpy])








if __name__ == '__main__':
    main()