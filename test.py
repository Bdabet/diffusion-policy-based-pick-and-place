import rtde_control
import rtde_receive
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
from multiprocessing.managers import SharedMemoryManager


from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer

shm_manager = SharedMemoryManager()
shm_manager.start()

max_pos_speed=0.25
max_rot_speed=0.2
cube_diag = np.linalg.norm([1,1,1])
tcp_offset=0.13
j_init = np.array([0,-90,-90,-90,90,0]) / 180 * np.pi
max_obs_buffer_size=30

robot = RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip="134.28.40.74",
            frequency=500, # UR5 CB3 RTDE #ur10e accroding to website 
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            launch_timeout=3,
            tcp_offset_pose=[0,0,tcp_offset,0,0,0],
            payload_mass=None,
            payload_cog=None,
            joints_init=j_init,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size
)
robot.start(wait=False)
print(robot.is_ready)





# # realsense
# camera_serial_numbers = SingleRealsense.get_connected_devices_serial()
# print(camera_serial_numbers)


# # initialiuzations
# video_capture_fps=30
# video_capture_resolution=(1280,720)
# obs_image_resolution=(640,480)
# obs_float32=False
# multi_cam_vis_resolution=(1280,720)
# record_raw_video=True
# frequency=10
# video_crf=21
# thread_per_video=2
# max_obs_buffer_size=30
# enable_multi_cam_vis=True,



# # camera relevant functions
# color_tf = get_image_transform(
#             input_res=video_capture_resolution,
#             output_res=obs_image_resolution, 
#             # obs output rgb
#             bgr_to_rgb=True)
# color_transform = color_tf
# if obs_float32:
#     color_transform = lambda x: color_tf(x).astype(np.float32) / 255

# def transform(data):
#         data['color'] = color_transform(data['color'])
#         return data

# rw, rh, col, row = optimal_row_cols(
#     n_cameras=len(camera_serial_numbers),
#     in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
#     max_resolution=multi_cam_vis_resolution
# )
# vis_color_transform = get_image_transform(
#     input_res=video_capture_resolution,
#     output_res=(rw,rh),
#     bgr_to_rgb=False
# )
# def vis_transform(data):
#     data['color'] = vis_color_transform(data['color'])
#     return data

# recording_transfrom = None
# recording_fps = video_capture_fps
# recording_pix_fmt = 'bgr24'
# if not record_raw_video:
#     recording_transfrom = transform
#     recording_fps = frequency
#     recording_pix_fmt = 'rgb24'

# video_recorder = VideoRecorder.create_h264(
#     fps=recording_fps, 
#     codec='h264',
#     input_pix_fmt=recording_pix_fmt, 
#     crf=video_crf,
#     thread_type='FRAME',
#     thread_count=thread_per_video)

# realsense = MultiRealsense(
#     serial_numbers=camera_serial_numbers,
#     shm_manager=shm_manager,
#     resolution=video_capture_resolution,
#     capture_fps=video_capture_fps,
#     put_fps=video_capture_fps,
#     # send every frame immediately after arrival
#     # ignores put_fps
#     put_downsample=False,
#     record_fps=recording_fps,
#     enable_color=True,
#     enable_depth=False,
#     enable_infrared=False,
#     get_max_k=max_obs_buffer_size,
#     transform=transform,
#     vis_transform=vis_transform,
#     recording_transform=recording_transfrom,
#     video_recorder=video_recorder,
#     verbose=False
#     )

# multi_cam_vis = None
# if enable_multi_cam_vis:
#     multi_cam_vis = MultiCameraVisualizer(
#         realsense=realsense,
#         row=row,
#         col=col,
#         rgb_to_bgr=False
#     )


# realsense.start(wait=False)
# print(realsense.is_ready)


