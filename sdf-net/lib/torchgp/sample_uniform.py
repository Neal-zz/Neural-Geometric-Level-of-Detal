import torch

def sample_uniform(num_samples : int):
    """
    Args:
        num_samples(int) : number of points to sample (100k)
    """
    return torch.rand(num_samples, 3) * 2.0 - 1.0

