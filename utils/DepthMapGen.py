import os
import os.path as osp
import time
import torch
import cv2
import subprocess
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import DictConfig
import hydra

def run_executable_subprocess(target, args=None):
    """
    Returns: tuple (return_code, stdout, stderr)
    """
    # Prepare command with any arguments
    cmd = [target]
    if args:
        if isinstance(args, str):
            cmd.append(args)
        elif isinstance(args, list):
            cmd.extend(args)
    
    try:
        # Run the process and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True  # Return strings instead of bytes
        )
        
        # Get output and error (if any)
        stdout, stderr = process.communicate()
        
        return process.returncode, stdout, stderr
    
    except FileNotFoundError:
        return -1, "", "Error: a.out not found in current directory"
    except PermissionError:
        return -1, "", "Error: Permission denied when trying to execute a.out"
    except Exception as e:
        return -1, "", f"Error: {str(e)}"

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function for depth estimation using PatchFusion.
    
    Args:
        cfg: Hydra configuration object
    """
    depthMapGenConf = cfg.utils.DepthMapGen
    
    args = []
    args.append(depthMapGenConf.PatchFusion_target)
    for key, val in depthMapGenConf.items():
        if key == 'PatchFusion_target' or key == 'PatchFusion_path':
            continue
        if key == 'config':
            args.append(str(val))
        elif key == 'image-raw-shape' or key == 'patch-split-num':
            args.append(f"--{key}")
            shape = depthMapGenConf[key].split(' ')
            args.append(f"{shape[0]}")
            args.append(f"{shape[1]}")
        elif not isinstance(val, bool):
            args.append(f"--{key}")
            args.append(str(val))
        elif val:
            args.append(f"--{key}")

    
    print(args)
    cwd = os.getcwd()
    os.chdir(depthMapGenConf.PatchFusion_path)
    ret_code, stdout, stderr = run_executable_subprocess('python', args)
    os.chdir(cwd)


    print(f"Return code: {ret_code}")
    print(f"Output: {stdout}")
    print(f"Error: {stderr}")

if __name__ == '__main__':
    main()