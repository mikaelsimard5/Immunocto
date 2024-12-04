import argparse

def parse_arguments():
    """
    Parse command-line arguments for specifying GPUs and a configuration file.

    Returns:
        tuple: A tuple containing:
            - config (str): Path to the configuration file.
            - n_gpus (int): Number of GPUs to use.
            - gpu_devices (list): List of GPU device IDs.
    """
    parser = argparse.ArgumentParser(description="Simple GPU and config argument parser.")
    
    parser.add_argument(
        '--gpus',
        type=eval,
        default=1,
        help="Number of GPUs as an integer (e.g., 1) or a list of devices (e.g., [0, 1]). Default: 1."
    )
    parser.add_argument(
        '--config',
        type=str,
        default="",
        help="Path to the configuration file. Default: an empty string."
    )
    
    args = parser.parse_args()
    
    # Validate and process GPU input
    try:
        if isinstance(args.gpus, int):
            n_gpus = args.gpus
            gpu_devices = list(range(n_gpus))
        elif isinstance(args.gpus, list):
            n_gpus = len(args.gpus)
            gpu_devices = args.gpus
        else:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError(
            "--gpus must be an integer (e.g., 1) or a list of integers (e.g., [0, 1])."
        )

    return args.config, n_gpus, gpu_devices
