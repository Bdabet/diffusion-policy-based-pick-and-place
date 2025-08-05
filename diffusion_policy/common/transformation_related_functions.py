    
from scipy.spatial.transform import Rotation as R
import numpy as np



def rotate_around_local_z(grasping_pose, rotation_angle):
    
        # Current rotation based on the grasping pose
        r_current = R.from_euler('xyz', grasping_pose[3:], degrees=False)

        
        r_local_z = R.from_euler('z', np.deg2rad(rotation_angle), degrees=False)
        
            

        # Combine the rotations
        r_new = r_current * r_local_z

        # debug 
        print("r_new", r_new.as_euler('xyz', degrees=True))

        # Update the final grasping pose
        grasping_pose = np.concatenate((grasping_pose[:3], r_new.as_euler('xyz', degrees=False)), axis=None)

        return grasping_pose