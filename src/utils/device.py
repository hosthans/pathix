import torch

def configure_device(config: dict = None) -> torch.device:
    """Return the configured CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():

        if config and 'gpu_id' in config:
            torch.cuda.set_device(config['gpu_id'])
            return torch.device(f'cuda:{config["gpu_id"]}')
        return torch.device('cuda')
    return torch.device('cpu')