#!/usr/bin/env python3
# infer.py --------------------------------------------------------------------
"""
Query a trained network on arbitrary points.

Example:
$ python infer.py --ckpt checkpoints/sdfmlp_epoch0120_val0.000096.pt \
                  --coords 0 0 0  1 0 0
"""
import argparse, torch, numpy as np
from .model import SDFMLP

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   required=True, help="Checkpoint .pt from train.py")
    p.add_argument("--coords", nargs="+", type=float, required=True,
                   help="Flat list of xyz coordinates; length must be multiple of 3.")
    p.add_argument("--device", default="cpu", help="cuda | cpu | cuda:1 ...")
    p.add_argument("--sdf-gp-fun", default="sdf_sphere", help="function to use for sdf")
    return p.parse_args()

import nsdf.example_sdf as example_sdf
from mmcore.geom.implicit.mc import *
def mc(fun, root:AABB, fname:str=None):
    #root = AABB((0, 0, 0), (1.5, 1.5, 1.5))  # anisotropic root volume
    fname=fname or fun.__name__
    kept, visited = subdivide(root, fun, max_depth=7)
    print(f"Tree: visited {visited:,d}, kept {kept:,d} leaves")
    import time
    
    s = time.perf_counter()
    verts, faces = {}, []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.children:
            stack.extend(n.children)
        else:
            polygonise_leaf(n, fun, verts, faces, preserve_orientation=True)
    print(time.perf_counter() - s)
    
    print(f"Mesh: {len(verts):,d} vertices, {len(faces):,d} triangles")
    write_ply(f"infer-{fname}.ply", verts, faces)
    print(f"Output written to  infer-{fname}.ply")
    
def make_neural_sdf(model):
    def neural_sdf_eval(pts):
        with torch.no_grad():
          
            res=model(torch.from_numpy(np.atleast_2d(pts).astype(np.float32)).cpu()   ).numpy()
        if res.shape[0]==1:
            return res[0]
        else:
            return res
    return neural_sdf_eval
    
    
    
    
def main():
    args = parse_args()
    device = torch.device(args.device if args.device else "cpu")
    pts    = np.asarray(args.coords, dtype=np.float32).reshape(-1,3)
    pts_t  = torch.from_numpy(pts).to(device)

    model = SDFMLP().to(device)
    state = torch.load(args.ckpt, map_location=device)["model"]
    model.load_state_dict(state)
    model.eval()
    
    with torch.no_grad():
        sdf = model(pts_t.cpu()).numpy()
    gp_sdf=getattr(example_sdf, args.sdf_gp_fun)
    resl_sdf = gp_sdf(pts_t.cpu().numpy())

    for xyz, d, real_d in zip(pts, sdf, resl_sdf):
        print(f"SDF({xyz}): {d:+.6f};  GP: {real_d:+.6f}")
    nsdf=make_neural_sdf(model)
    mc(gp_sdf, AABB((0, 0, 0), (1.5, 1.5, 1.5)), fname=f'analytic-{args.sdf_gp_fun}.ply')
    mc(nsdf,AABB((0, 0, 0), (1.5, 1.5, 1.5)), fname=f'neural-{args.sdf_gp_fun}.ply')
   
if __name__ == "__main__":
    main()
