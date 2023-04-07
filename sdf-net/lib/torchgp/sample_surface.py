import torch
from .random_face import random_face
from .area_weighted_distribution import area_weighted_distribution

def sample_surface(V : torch.Tensor, F : torch.Tensor, num_samples : int, distrib = None):
    """
    Args:
        V (torch.Tensor): [n,3] array of vertices
        F (torch.Tensor): [n,3] array of indices
        num_samples (int): number of surface samples (100k)
        distrib: 以三角形面积大小为概率进行采样
    """
    if distrib is None:
        distrib = area_weighted_distribution(V, F)

    # 对三角形采样
    fidx, normals = random_face(V, F, num_samples, distrib)
    f = V[fidx]
    # [num_samples,1] 0-1 u 应该偏大一些
    u = torch.sqrt(torch.rand(num_samples)).to(V.device).unsqueeze(-1)
    # [num_samples,1] 0-1
    v = torch.rand(num_samples).to(V.device).unsqueeze(-1)
    # 在三角形内部随机选择一个点 [num_samples,3ind,3xyz]
    samples = (1 - u) * f[:,0,:] + (u * (1 - v)) * f[:,1,:] + u * v * f[:,2,:]
    return samples, normals

