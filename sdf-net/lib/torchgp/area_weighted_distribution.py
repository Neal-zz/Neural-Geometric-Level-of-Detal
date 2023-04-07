import torch
from .per_face_normals import per_face_normals

def area_weighted_distribution(V : torch.Tensor, F : torch.Tensor, normals : torch.Tensor = None):
    """
    Args:
        V (torch.Tensor): [n,3] array of vertices
        F (torch.Tensor): [n,3] array of indices
        normals (torch.Tensor): normals (if precomputed)
    """

    # None
    if normals is None:
        # [n,3]
        normals = per_face_normals(V, F)
    # 三角形面积 [n]
    areas = torch.norm(normals, p=2, dim=1) * 0.5
    # 令面积之和为 1
    areas /= torch.sum(areas) + 1e-10
    # 以面积大小为概率进行采样
    return torch.distributions.Categorical(areas.view(-1))

