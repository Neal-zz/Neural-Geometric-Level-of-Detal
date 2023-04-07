import torch
from torch.utils.data import Dataset

from lib.torchgp import load_obj, normalize, point_sample, compute_sdf, sample_surface
from lib.utils import setparam

class MeshDataset(Dataset):
    """Base class for single mesh datasets."""

    def __init__(self, 
        args = None, 
        dataset_path = None,
        raw_obj_path = None,
        sample_mode = None,
        get_normals = None,
        seed = None,
        num_samples = None,
        trim = None,
        sample_tex = None
    ):
        self.args = args
        # data/armadillo.obj
        self.dataset_path = setparam(args, dataset_path, 'dataset_path')
        # None
        self.raw_obj_path = setparam(args, raw_obj_path, 'raw_obj_path')
        # ['rand', 'near', 'near', 'trace', 'trace']
        self.sample_mode = setparam(args, sample_mode, 'sample_mode')
        # False
        self.get_normals = setparam(args, get_normals, 'get_normals')
        # 100k
        self.num_samples = setparam(args, num_samples, 'num_samples')
        # False
        self.trim = setparam(args, trim, 'trim')
        # False
        self.sample_tex = setparam(args, sample_tex, 'sample_tex')
        if self.sample_tex:
            out = load_obj(self.dataset_path, load_materials=True)
            self.V, self.F, self.texv, self.texf, self.mats = out
        else:
            # vertices [n,3] and faces [n,3]
            self.V, self.F = load_obj(self.dataset_path)
        # 将模型移到原点，并缩放到 +-1
        self.V, self.F = normalize(self.V, self.F)
        self.mesh = self.V[self.F]
        self.resample()

    def resample(self):
        """Resample SDF samples."""
        self.nrm = None
        # False
        if self.get_normals:
            self.pts, self.nrm = sample_surface(self.V, self.F, self.num_samples*5)
            self.nrm = self.nrm.cpu()
        else:
            # 用五种方法共采样 500k 个点云
            self.pts = point_sample(self.V, self.F, self.sample_mode, self.num_samples)
        # [n,3] -> [n] sdf
        self.d = compute_sdf(self.V.cuda(), self.F.cuda(), self.pts.cuda())   
        self.d = self.d[...,None]
        self.d = self.d.cpu()
        self.pts = self.pts.cpu()

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        # False
        if self.get_normals:
            return self.pts[idx], self.d[idx], self.nrm[idx]
        # False
        elif self.sample_tex:
            return self.pts[idx], self.d[idx], self.rgb[idx]
        else:
            return self.pts[idx], self.d[idx]
            
    def __len__(self):
        """Return length of dataset (number of _samples_)."""
        return self.pts.size()[0]

    def num_shapes(self):
        """Return number of _mesh models_."""
        return 1
