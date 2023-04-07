import numpy as np
import torch
from lib.renderer import SphereTracer
from lib.geoutils import sample_unif_sphere, voxel_corners
from lib.utils import PerfTimer

# Utils that use rendering features

def sample_surface(n, net, sol=False, device='cuda'):
    
    timer = PerfTimer(activate=False)
    # tracer = SphereTracer(device, sol=sol) 报错
    tracer = SphereTracer(device, sol)

    # Sample surface using random tracing (resample until num_samples is reached)
    i = 0
    while i < 1000:
        ray_o = torch.rand((n, 3), device=device) * 2.0 - 1.0
        # this really should just return a torch array in the first place
        ray_d = torch.from_numpy(sample_unif_sphere(n)).float().to(device)
        
        rb = tracer(net, ray_o, ray_d)
        pts = rb.x
        hit = rb.hit

        pts_pr = pts[hit] if i == 0 else torch.cat([pts_pr, pts[hit]], dim=0)
        if pts_pr.shape[0] >= n:
            break
    
        i += 1
        if i == 50:
            print('Taking an unusually long time to sample desired # of points.')
    timer.check(f"done in {i} iterations")

    return pts_pr

def voxel_sparsify(n, net, lod, sol=False, device='cuda'):
    
    #lod = 5

    _lod = net.lod
    net.lod = lod
    # surface = sample_surface(n, net, sol=sol, device=device)[:n] 报错
    surface = sample_surface(n, net, sol, device)[:n]
    vs = []

    for i in range(lod+1):
        res = 2 ** (i+2)
        uniq = torch.unique( ( ((surface+1.0) / 2.0) * res).floor().long(), dim=0)
        vs.append(uniq)

    net.lod = _lod
    return vs

