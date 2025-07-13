#!/usr/bin/env python3
# infer.py --------------------------------------------------------------------
"""
Query a trained network on arbitrary points.

Example:
$ python infer.py --ckpt checkpoints/sdfmlp_epoch0120_val0.000096.pt \
                  --coords 0 0 0  1 0 0
"""
import argparse, torch, numpy as np
from model import SDFMLP

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",   required=True, help="Checkpoint .pt from train.py")
    p.add_argument("--coords", nargs="+", type=float, required=True,
                   help="Flat list of xyz coordinates; length must be multiple of 3.")
    p.add_argument("--device", default="cuda", help="cuda | cpu | cuda:1 ...")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pts    = np.asarray(args.coords, dtype=np.float32).reshape(-1,3)
    pts_t  = torch.from_numpy(pts).to(device)

    model = SDFMLP().to(device)
    state = torch.load(args.ckpt, map_location=device)["model"]
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        sdf = model(pts_t).cpu().numpy()

    for xyz, d in zip(pts, sdf):
        print(f"SDF({xyz}) = {d:+.6f}")

if __name__ == "__main__":
    main()
