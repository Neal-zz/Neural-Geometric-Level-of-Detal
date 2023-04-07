import torch
from .sample_surface import sample_surface
from .area_weighted_distribution import area_weighted_distribution

def sample_near_surface(V : torch.Tensor, F : torch.Tensor, num_samples: int,
    variance : float = 0.01, distrib=None):
    """
    Args:
        V (torch.Tensor): [n,3] array of vertices
        F (torch.Tensor): [n,3] array of indices
        num_samples (int): number of surface samples (100k)
        distrib: 以三角形面积大小为概率进行采样
    """
    if distrib is None:
        distrib = area_weighted_distribution(V, F)
    # [num_samples,3ind,3xyz]
    samples = sample_surface(V, F, num_samples, distrib)[0]
    # 在 sample_surface 基础上增加 0.01 尺度的偏移
    samples += torch.randn_like(samples) * variance
    return samples
