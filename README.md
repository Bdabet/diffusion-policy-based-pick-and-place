# Diffusion Policy – IFPT Adapted Implementation  
### Modified for UR10e Pick-and-Place, Multi-Object Tasks, Text Conditioning, and Real-World Execution

This repository contains an extended and hardware-adapted version of the official **Diffusion Policy** framework by Chi et al. (2023–2024).  
It enables real-world imitation learning on a **UR10e robot** equipped with a **Zimmer GEP2013IO-00-A** gripper, multiple **Intel RealSense cameras**, and an **Xbox controller teleoperation** interface for demonstration collection.

The code has been adapted to support:

- **6-DoF end-effector control** (including vertical motion and yaw rotation)
- **Gripper control** via UR digital outputs
- **Text-conditioned imitation learning**
- **Image-conditioned imitation learning** 
- **Multi-object sequential pick-and-place**
 

## Results

The adapted diffusion policy achieves reliable real-world performance across a range of UR10e pick-and-place tasks, including multi-object and in-box scenarios. Policies are consistently selected at the earliest low-loss checkpoint to avoid overfitting, which often degrades success rates at later epochs.

### Main task performance

**Basic pick-and-place (single block)**  
Best checkpoint reaches an average success score of **72.00%** over 50 trials (36 successes, 14 failures).  
Primary failure modes include off-centered grasps and premature gripper closure. Extending training to 1000 epochs reduces success to **57.00%**, indicating overfitting despite similar validation loss.  
**Takeaway:** Performance is primarily limited by grasp reliability, and later training epochs lead to overfitting.

---

**Text-conditioned pick-and-place (3 goals)**  
The best checkpoint achieves **86.67%** success over 60 trials. The policy consistently places the object at the correct conditioned target and never at an incorrect goal location.  
Failures are dominated by grasp misalignment and timeouts rather than incorrect goal interpretation.  
**Takeaway:** Text conditioning enables reliable goal disambiguation, with failures driven by grasp execution.

---

**In-box conditioned pick-and-place (1 object in movable box)**  
Success reaches **61.67%** at the best checkpoint. Dynamic box poses and tighter spatial constraints significantly increase task difficulty compared to the table-top setup.  
**Takeaway:** Dynamic containers and constrained workspaces substantially increase task difficulty.

---

**Multi-object conditioned pick-and-place (3 objects, table-top)**  
With three sequential pick-and-place actions per episode, the best model achieves **94.44%** success over 66 trials. Performance benefits from longer-horizon training and a larger effective number of demonstrated sub-trajectories.  
**Takeaway:** Multi-object training benefits from longer horizons and increased demonstration coverage.

---

**In-box multi-object conditioned pick-and-place (3 objects in box)**  
The best checkpoint attains **84.67%** success over 50 trials. Performance drops moderately relative to the table-top multi-object task while maintaining robust multi-goal execution.  
**Takeaway:** Box constraints cause a moderate performance drop while preserving multi-goal robustness.

---

### Ablations and generalization

**Demonstration quality**  
With identical dataset sizes, a policy trained on human demonstrations reaches **90%** success, compared to **62%** for a scripted dataset.  
**Takeaway:** Richer human demonstrations substantially improve generalization.

---

**Dataset size**  
Reducing scripted demonstrations from 220 to 75 episodes lowers success from **72.00%** to **62.00%** on the basic task.  
**Takeaway:** Success rates scale positively with dataset size.

---

**Camera views**  
Adding a third RealSense view reduces success from **86.67%** to **63.64%** in the text-conditioned task.  
**Takeaway:** Additional viewpoints can introduce harmful noise if they do not add task-relevant information.

---

**Conditioning modality**  
Replacing text conditioning with image conditioning reduces success from **86.67%** to **43.44%** and increases partial successes.  
**Takeaway:** Text-based conditioning is currently more reliable than image-based goals.

---

**Unseen objects**  
Generalization to unseen green rectangular and semicircular blocks reaches **77.50%** and **67.50%**, respectively, while performance drops significantly for unseen colors and curved objects.  
**Takeaway:** Generalization is stronger to familiar geometries than to novel shapes or colors.

---

**Fine-tuning to a new goal**  
Adding a fourth conditioned goal via fine-tuning yields **81.90%** overall success, with the new goal reaching **82.6%**. Training time is reduced from approximately 42 hours to 16 hours.  
**Takeaway:** Fine-tuning enables efficient adaptation with limited forgetting.


### Example Episodes: Experiment 4

<p align="center">
  <img src="media\sucess_sample.gif" width="340">
  <img src="media\failure_sample.gif" width="340">
</p>

<p align="center">
  <em>Left: Successful multi-object pick-and-place episode. Right: Failure due to off-centered grasp or premature gripper closure.</em>
</p>


## Citation

If you use this repository in your research, please cite:

Chi, C., Wang, Y., Hafner, D., Hausman, K, Driess, D., & Florence, P. (2023).
Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.
Robotics: Science and Systems (RSS).

@inproceedings{chi2023diffusionpolicy,
title = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
author = {Chi, Cheng and Wang, Yiming and Hafner, Danijar and Hausman, Karol and Driess, Danny and Florence, Pete},
booktitle = {Robotics: Science and Systems (RSS)},
year = {2023}
}

