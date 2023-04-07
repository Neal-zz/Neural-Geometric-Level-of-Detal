import torch
from .area_weighted_distribution import area_weighted_distribution
from .per_face_normals import per_face_normals

def random_face(V : torch.Tensor, F : torch.Tensor, num_samples : int, distrib=None):
    """
    Args:
        V (torch.Tensor): [n,3] array of vertices
        F (torch.Tensor): [n,3] array of indices
        num_samples (int): num of samples to return (100k)
        distrib: 以三角形面积大小为概率进行采样
    """
    if distrib is None:
        distrib = area_weighted_distribution(V, F)
    # [n,3]
    normals = per_face_normals(V, F)
    idx = distrib.sample([num_samples])
    # 对三角形采样
    return F[idx], normals[idx]

