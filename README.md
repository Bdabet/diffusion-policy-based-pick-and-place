# Diffusion Policy – IFPT Adapted Implementation  
### Modified for UR10e Pick-and-Place, Multi-Object Tasks, Text Conditioning, and Real-World Execution

This repository contains an extended and hardware-adapted version of the official **Diffusion Policy** framework by Chi et al. (2023–2024).  
It enables real-world imitation learning on a **UR10e robot** equipped with a **Zimmer GEP2013IO-00-A** gripper, multiple **Intel RealSense cameras**, and an **Xbox controller teleoperation** interface for demonstration collection.

The code has been adapted to support:

- **6-DoF end-effector control** (including vertical motion and yaw rotation)
- **Gripper control** via UR digital outputs
- **Text-conditioned imitation learning**
- **Image-conditioned imitation learning** (movable box)
- **Multi-object sequential pick-and-place**
 

 ## Citation

If you use this repository, please cite the original Diffusion Policy paper:

**Chi, C., Wang, Y., Hafner, D., Hausman, K., Driess, D., & Florence, P. (2023).  
Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.  
Robotics: Science and Systems (RSS).**

BibTeX:
@inproceedings{chi2023diffusionpolicy,
  title={Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  author={Chi, Cheng and Wang, Yiming and Hafner, Danijar and Hausman, Karol and Driess, Danny and Florence, Pete},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2023}
}

