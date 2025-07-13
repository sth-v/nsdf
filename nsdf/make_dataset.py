#!/usr/bin/env python3
# make_dataset.py -------------------------------------------------------------
"""
Generate a training set by sampling an analytic SDF.

Example (use the unit sphere above):
$ python make_dataset.py \
        --sdf-module example_sdf --sdf-fn sdf \
        --bbox -1 1 -1 1 -1 1 \
        --n-uniform 800000 \
        --n-surface 200000 --surface-eps 0.005 \
        --out data_sphere.npz
"""

import argparse, importlib, sys, pathlib
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sdf-module", required=True,
                   help="Python module path that contains the SDF function.")
    p.add_argument("--sdf-fn",     default="sdf",
                   help="Function name inside the module.")
    p.add_argument("--bbox",       nargs=6, type=float, required=True,
                   metavar=("xmin","xmax","ymin","ymax","zmin","zmax"),
                   help="Axis‑aligned bounding box for sampling.")
    p.add_argument("--n-uniform",  type=int, default=1_000_000,
                   help="# uniform samples in the whole volume.")
    p.add_argument("--n-surface",  type=int, default=200_000,
                   help="# near‑surface samples (|sdf| < eps).")
    p.add_argument("--surface-eps",type=float, default=0.005,
                   help="Half‑thickness of the near‑surface shell.")
    p.add_argument("--out",        required=True,
                   help="Output .npz file.")
    return p.parse_args()

def main():
    args = parse_args()
    mod = importlib.import_module(args.sdf_module)
    sdf_fn = getattr(mod, args.sdf_fn)

    bbox = np.array(args.bbox, dtype=np.float32).reshape(3,2)
    rng  = np.random.default_rng()

    # 1. uniform volume samples ------------------------------------------------
    pts_uni = rng.uniform(bbox[:,0], bbox[:,1], size=(args.n_uniform, 3)).astype(np.float32)
    sdf_uni = sdf_fn(pts_uni).astype(np.float32)

    # 2. near‑surface samples --------------------------------------------------
    # oversample, then filter by |sdf| < eps to guarantee enough points
    oversample = int(args.n_surface * 2.5)
    pts_over   = rng.uniform(bbox[:,0], bbox[:,1], size=(oversample, 3)).astype(np.float32)
    sdf_over   = sdf_fn(pts_over).astype(np.float32)
    mask       = np.abs(sdf_over) < args.surface_eps
    if mask.sum() < args.n_surface:
        print(f"[make_dataset] Increase --surface-eps or oversample size; "
              f"only {mask.sum()} points in shell.", file=sys.stderr)
        sys.exit(1)
    pts_surf, sdf_surf = pts_over[mask][:args.n_surface], sdf_over[mask][:args.n_surface]

    # 3. concatenate & shuffle -------------------------------------------------
    pts   = np.concatenate([pts_uni, pts_surf], axis=0)
    sdf   = np.concatenate([sdf_uni, sdf_surf], axis=0)
    idx   = rng.permutation(len(pts))
    pts, sdf = pts[idx], sdf[idx]

    # 4. save ------------------------------------------------------------------
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, points=pts.astype(np.float32), sdf=sdf)
    print(f"[make_dataset] wrote {pts.shape[0]} samples → {args.out}")

if __name__ == "__main__":
    main()
