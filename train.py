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

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    
    # Define custom output directory
    # custom_output_dir = "/workspace/diffusion_policy/data/outputs/2025.07.10/13.09.43_train_diffusion_unet_image_real_image"
    
    
    
   

    # Initialize workspace with the custom output directory
    workspace: BaseWorkspace = cls(cfg)
    # workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
