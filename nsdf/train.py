#!/usr/bin/env python3
# train.py --------------------------------------------------------------------
"""
Example:
$ python train.py --data data_sphere.npz --outdir checkpoints
"""

import argparse, pathlib, time, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sdf_dataset import SDFDataset
from model import SDFMLP

# ------------------------ hyper‑parameters -----------------------------------
BATCH_SIZE      = 32768          # fits comfortably on 8 GB GPUs; adjust if needed
LEARNING_RATE   = 1e-4
EPOCHS          = 500
EIKONAL_WEIGHT  = 0.1
VALID_FRACTION  = 0.05

# ------------------------ utils ---------------------------------------------
def eikonal_loss(pred: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    pred   : (B,)
    coords : (B,3) with requires_grad=True
    returns: scalar
    """
    grads = torch.autograd.grad(pred,
                                coords,
                                grad_outputs=torch.ones_like(pred),
                                create_graph=True,
                                retain_graph=True,
                                only_inputs=True)[0]
    return ((grads.norm(dim=-1) - 1.0) ** 2).mean()

# ------------------------ main ----------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   required=True, help=".npz file from make_dataset.py")
    p.add_argument("--outdir", required=True, help="Directory to store checkpoints & logs")
    p.add_argument("--device", default="cpu", help="cuda | cpu | cuda:1 ...")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          else "cpu")

    # dataset & split ---------------------------------------------------------
    ds_full = SDFDataset(args.data)
    n_val   = int(len(ds_full) * VALID_FRACTION)
    ds_train, ds_val = random_split(ds_full, [len(ds_full)-n_val, n_val])
    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # model -------------------------------------------------------------------
    model  = SDFMLP().to(device)
    model  = torch.compile(model)           # PyTorch 2.x
    optim  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    outdir = pathlib.Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    # training loop -----------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, running = time.time(), 0.0
        for coords, gt in loader_train:
            coords, gt = coords.to(device), gt.to(device)
            coords.requires_grad_()
            with torch.cuda.amp.autocast():
                pred = model(coords)
                loss = F.mse_loss(pred, gt) + EIKONAL_WEIGHT * eikonal_loss(pred, coords)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            running += loss.item() * len(coords)

        train_loss = running / len(ds_train)

        # validation ----------------------------------------------------------
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            val_running = 0.0
            for coords, gt in loader_val:
                coords, gt = coords.to(device), gt.to(device)
                pred = model(coords)
                val_running += F.mse_loss(pred, gt, reduction="sum").item()
            val_loss = val_running / len(ds_val)

        # save checkpoint if better ------------------------------------------
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = outdir / f"sdfmlp_epoch{epoch:04d}_val{val_loss:.6f}.pt"
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss}, ckpt_path)
            flag = "✓"
        else:
            flag = " "

        print(f"[{epoch:4d}/{EPOCHS}] "
              f"train {train_loss:.6e} | val {val_loss:.6e} {flag} "
              f"| {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
