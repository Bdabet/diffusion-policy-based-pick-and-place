from diffusion_policy.real_world.single_realsense import *
from multiprocessing.managers import SharedMemoryManager
import time


sm = SharedMemoryManager()
sm.start()
print("SharedMemoryManager started.")

cameras = [SingleRealsense(sm, serial_number='051122072208')]#, SingleRealsense(sm, serial_number='936322071866')]
for cam in cameras:
    print("Starting camera with serial number:", cam.serial_number)
    put_start_time = time.time()
    cam.start(wait=False, put_start_time=put_start_time)
    print(f"Camera {cam.serial_number} started.")

while True:
    time.sleep(2)  # Allow some time for cameras to start
    for cam in cameras:
        print(cam.is_ready)