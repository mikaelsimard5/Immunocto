# Point of this code:
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Simple GPU and config argument parser.")
    
    parser.add_argument(
        '--gpus',
        type=eval,  # Accept either a scalar (int) or a list ([0,1,...])
        default=1,
        help="Number of GPUs as a scalar (e.g., 1) or a list of devices (e.g., [0,1]). Default: 1."
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default="",
        help="Path to the configuration file. Default: ""."
    )
    
    args = parser.parse_args()
    
    # Process GPU input
    if isinstance(args.gpus, int):
        n_gpus = args.gpus
        gpu_devices = list(range(n_gpus))
    elif isinstance(args.gpus, list):
        n_gpus = len(args.gpus)
        gpu_devices = args.gpus
    else:
        raise ValueError("Invalid value for --gpus. Must be a scalar (int) or a list.")
    
    return args.config, n_gpus, gpu_devices