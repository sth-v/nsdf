#!/usr/bin/env python3
# train.py (CPU/GPU friendly) --------------------------------------------------
"""
Example:
  # CPU‑only MacBook
  python train.py --data data_sphere.npz --device cpu --batch 8192 --no-compile

  # CUDA GPU (unchanged behaviour)
  python train.py --data data_sphere.npz --device cuda
"""

import argparse, pathlib, time, contextlib, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sdf_dataset import SDFDataset
from model import SDFMLP

# --------------------------------------------------------------------------- #
# Hyper‑parameters (can all be overridden on CLI)                             #
# --------------------------------------------------------------------------- #
DEFAULT_BATCH_GPU  = 32768
DEFAULT_BATCH_CPU  = 8192*2
LEARNING_RATE      = 1e-4
EPOCHS             = 500
EIKONAL_WEIGHT     = 0.1
VALID_FRACTION     = 0.05

# --------------------------------------------------------------------------- #
# Utility: dummy scaler works exactly like GradScaler, but no‑ops on CPU      #
# --------------------------------------------------------------------------- #
class DummyScaler:
    def scale(self, loss): return loss
    def step(self, optim): optim.step()
    def update(self): pass

# --------------------------------------------------------------------------- #
# Eikonal regulariser                                                         #
# --------------------------------------------------------------------------- #
def eikonal_loss(pred: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    grads = torch.autograd.grad(pred,
                                coords,
                                grad_outputs=torch.ones_like(pred),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True)[0]
    return ((grads.norm(dim=-1) - 1.0) ** 2).mean()

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True,  help=".npz file from make_dataset.py")
    p.add_argument("--outdir", required=True,  help="Where to save checkpoints")
    p.add_argument("--device", default="cuda", help="'cuda', 'cpu', or 'cuda:N'")
    p.add_argument("--batch",  type=int,       help="Batch size (override default)")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip torch.compile even on GPU (for Triton issues)")
    p.add_argument("--epochs", type=int, default=EPOCHS,)
    return p.parse_args()

# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    args   = parse_args()
    device = torch.device(args.device)
    is_cuda = device.type == "cuda"
    is_mps = device.type == "mps"
    is_gpu = is_cuda or is_mps
    batch_size = args.batch or (DEFAULT_BATCH_GPU if is_cuda else DEFAULT_BATCH_CPU)
    amp_ctx    = lambda *args,**kwargs:torch.amp.autocast(device.type, *args,**kwargs)
    scaler     = torch.amp.GradScaler(device.type)
    #amp_ctx = torch.cuda.amp.autocast if is_cuda else contextlib.nullcontext
    #scaler = torch.cuda.amp.GradScaler() if is_cuda else DummyScaler()
    # ----------------------- Dataset & DataLoader --------------------------- #
    ds_full = SDFDataset(args.data)
    n_val   = int(len(ds_full) * VALID_FRACTION)
    ds_train, ds_val = random_split(ds_full, [len(ds_full)-n_val, n_val])

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=4 ,
                              pin_memory=is_cuda)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=2,
                              pin_memory=is_cuda)

    # ----------------------- Model ----------------------------------------- #
    model = SDFMLP().to(device)
    if not args.no_compile:
        model = torch.compile(model)      # only safe when Triton / clang present

    optim  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    # ----------------------- Training loop --------------------------------- #
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        t0 = time.time()

        for coords, gt in loader_train:
            coords, gt = coords.to(device), gt.to(device)
            coords.requires_grad_()

            with amp_ctx():
                pred  = model(coords)
                loss  = F.mse_loss(pred, gt) + EIKONAL_WEIGHT * eikonal_loss(pred, coords)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            running += loss.item() * len(coords)

        train_loss = running / len(ds_train)

        # ----------------------- Validation -------------------------------- #
        model.eval()
        val_running = 0.0
        with torch.no_grad(), amp_ctx():
            for coords, gt in loader_val:
                coords, gt = coords.to(device), gt.to(device)
                pred       = model(coords)
                val_running += F.mse_loss(pred, gt, reduction="sum").item()
        val_loss = val_running / len(ds_val)

        # ----------------------- Checkpoint -------------------------------- #
        if val_loss < best_val:
            best_val = val_loss
            ckpt = outdir / f"sdfmlp_epoch{epoch:04d}_val{val_loss:.6f}.pt"
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss}, ckpt)
            flag = "✓"
        else:
            flag = " "

        elapsed = time.time() - t0
        print(f"[{epoch:4d}/{args.epochs}] "
              f"train {train_loss:.6e} | val {val_loss:.6e} {flag} | {elapsed:.1f}s "
              f"({device.type})")

if __name__ == "__main__":
    main()