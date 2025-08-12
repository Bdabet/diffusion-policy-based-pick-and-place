"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace

#last keyboard domo video #89
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import os

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    # config_path=str(pathlib.Path(__file__).parent.joinpath(
    #     'diffusion_policy','config'))   # if you run train.py from /workspace/diffusion_policy directory use this config_path
    config_path=str(pathlib.Path(__file__).parent.joinpath('config'))   # if you run train.py from /workspace directory use this config_path
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    
    # Define custom output directory
    # custom_output_dir = "/workspace/diffusion_policy/data/outputs/2025.08.05/15.47.01_train_diffusion_unet_image_pick_and_place"
    custom_output_dir = "/workspace/diffusion_policy/data/outputs/train_diffusion_unet_image_pick_and_place_insertion_pp"
    if not os.path.exists(custom_output_dir):
        os.makedirs(custom_output_dir)
    
    # Initialize workspace with the custom output directory
    workspace: BaseWorkspace = cls(cfg, custom_output_dir)
    # workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
