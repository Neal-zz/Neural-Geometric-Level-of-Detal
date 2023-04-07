import torch
from .sample_near_surface import sample_near_surface
from .sample_surface import sample_surface
from .sample_uniform import sample_uniform
from .area_weighted_distribution import area_weighted_distribution

def point_sample(V : torch.Tensor, F : torch.Tensor, techniques : list, num_samples : int):
    """
    Args:
        V (torch.Tensor): [n,3] array of vertices
        F (torch.Tensor): [n,3] array of indices
        techniques (list[str]): list of techniques to sample with
        num_samples (int): points to sample per technique (100k)
    """

    # ['rand', 'near', 'near', 'trace', 'trace']
    if 'trace' in techniques or 'near' in techniques:
        # Precompute face distribution
        distrib = area_weighted_distribution(V, F)

    samples = []
    for technique in techniques:
        if technique =='trace':
            # 先以三角形面积大小为概率采样 100k 个三角形；再在三角形内部随机采样一个点
            samples.append(sample_surface(V, F, num_samples, distrib=distrib)[0])
        elif technique == 'near':
            # 在 sample_surface 基础上增加 0.01 尺度的偏移
            samples.append(sample_near_surface(V, F, num_samples, distrib=distrib))
        elif technique == 'rand':
            # 在 [-1,1] 立方体内随机采样
            samples.append(sample_uniform(num_samples).to(V.device))
    samples = torch.cat(samples, dim=0)
    return samples

