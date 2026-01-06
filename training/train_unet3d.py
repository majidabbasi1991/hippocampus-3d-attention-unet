# ============================================================
# DilatedConv + AttentionGate
# ============================================================

import os, json, math, time, random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- cuDNN settings: پایدار روی GTX1080 ---
import torch.backends.cudnn as cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# --- Dataset loading ---
from dataset_patches import (
    PREPROC_IMG_DIR, PREPROC_MSK_DIR, SPLIT_JSON,
    collect_pairs, save_or_load_split, HippocampusDataset
)

# ============================================================
# Global Settings
# ============================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIXED_PRECISION = False        # روی Pascal (GTX1080) AMP توصیه نمی‌شود
PATCH_SIZE = (80, 80, 80)
BATCH_SIZE = 1
NUM_WORKERS = 0

MAX_EPOCHS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
FG_RATIO = 0.7
VAL_EVERY = 1

SAVE_DIR = Path("E:/unet3d_runs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEST_CKPT = SAVE_DIR / "best_unet3d.pth"


# ============================================================
# Utility: reproducibility
# ============================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Model Components (Dilated DoubleConv + Attention Gate)
# ============================================================

# -----------------------------
# DoubleConv with dilation=2
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


# -----------------------------
# Attention Gate
# -----------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(F_int, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi


# -----------------------------
# UNet3D with Attention + Dilation
# -----------------------------
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=32):
        super().__init__()

        b = base_ch

        # ----- Encoder -----
        self.enc1 = DoubleConv(in_channels, b)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(b, b*2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv(b*2, b*4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = DoubleConv(b*4, b*8)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(b*8, b*16)

        # ----- Attention Gates -----
        self.att4 = AttentionGate(F_g=b*8, F_l=b*8, F_int=b*4)
        self.att3 = AttentionGate(F_g=b*4, F_l=b*4, F_int=b*2)
        self.att2 = AttentionGate(F_g=b*2, F_l=b*2, F_int=b)
        self.att1 = AttentionGate(F_g=b,   F_l=b,   F_int=b//2)

        # ----- Decoder -----
        self.up4 = nn.ConvTranspose3d(b*16, b*8, 2, stride=2)
        self.dec4 = DoubleConv(b*16, b*8)

        self.up3 = nn.ConvTranspose3d(b*8, b*4, 2, stride=2)
        self.dec3 = DoubleConv(b*8, b*4)

        self.up2 = nn.ConvTranspose3d(b*4, b*2, 2, stride=2)
        self.dec2 = DoubleConv(b*4, b*2)

        self.up1 = nn.ConvTranspose3d(b*2, b, 2, stride=2)
        self.dec1 = DoubleConv(b*2, b)

        self.outc = nn.Conv3d(b, out_channels, 1)

    def forward(self, x):
        # ----- Encoder -----
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        bn = self.bottleneck(self.pool4(e4))

        # ----- Decoder with Attention -----
        d4 = self.up4(bn)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return self.outc(d1)


# ============================================================
# Metrics and Loss
# ============================================================

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    pred = pred.float()
    target = target.float()
    inter = torch.sum(pred * target, dim=(2,3,4))
    union = torch.sum(pred, dim=(2,3,4)) + torch.sum(target, dim=(2,3,4))
    return ((2*inter + eps) / (union + eps)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bw = bce_weight
        self.dw = dice_weight

    def forward(self, logits, target):
        bce = self.bce(logits, target.float())
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        dice = 1.0 - dice_coefficient(preds, target)
        return self.bw * bce + self.dw * dice


# ============================================================
# Training / Validation / Test Loops
# ============================================================

def train_one_epoch(model, loader, optimizer, scaler, loss_fn):
    model.train()
    total_loss, n = 0.0, 0

    for batch in loader:
        img = batch["image"].to(DEVICE)
        msk = batch["mask"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        logits = model(img)
        loss = loss_fn(logits, msk)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    return total_loss / max(1, n)


@torch.no_grad()
def validate(model, loader, loss_fn):
    model.eval()
    total_loss, total_dice, n = 0.0, 0.0, 0

    for batch in loader:
        img = batch["image"].to(DEVICE)
        msk = batch["mask"].to(DEVICE)

        logits = model(img)
        loss = loss_fn(logits, msk)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        dice = dice_coefficient(preds, msk)

        total_loss += loss.item()
        total_dice += dice.item()
        n += 1

    return total_loss / max(1, n), total_dice / max(1, n)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_dice, n = 0.0, 0

    for batch in loader:
        img = batch["image"].to(DEVICE)
        msk = batch["mask"].to(DEVICE)

        logits = model(img)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        dice = dice_coefficient(preds, msk)

        total_dice += dice.item()
        n += 1

    return total_dice / max(1, n)


# ============================================================
# Main
# ============================================================

def main():
    set_seed(SEED)

    # 1) dataset split
    pairs = collect_pairs(PREPROC_IMG_DIR, PREPROC_MSK_DIR)
    split = save_or_load_split(pairs, SPLIT_JSON)

    # 2) datasets
    ds_train = HippocampusDataset(pairs, split["train"], mode="patch",
                                  patch_size=PATCH_SIZE, fg_ratio=FG_RATIO)
    ds_val   = HippocampusDataset(pairs, split["val"], mode="patch",
                                  patch_size=PATCH_SIZE, fg_ratio=FG_RATIO)
    ds_test  = HippocampusDataset(pairs, split["test"], mode="full")

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    dl_val   = DataLoader(ds_val, batch_size=1, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
    dl_test  = DataLoader(ds_test, batch_size=1, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    # 3) model/loss/optimizer
    model = UNet3D(in_channels=1, out_channels=1, base_ch=32).to(DEVICE)
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    best_val = -1
    print(f"Device: {DEVICE} | Patch: {PATCH_SIZE}")

    # 4) training loop
    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, dl_train, optimizer, None, loss_fn)
        msg = f"[Epoch {epoch:03d}] train_loss={train_loss:.4f}"

        val_loss, val_dice = validate(model, dl_val, loss_fn)
        msg += f" | val_loss={val_loss:.4f} val_dice={val_dice:.4f}"

        # save best
        if val_dice > best_val:
            best_val = val_dice
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_dice": best_val
            }, BEST_CKPT)
            msg += "  <-- saved best"

        print(msg + f" | time={time.time()-t0:.1f}s")

    print("\nTraining finished. Best val dice:", best_val)

    # 5) test
    if BEST_CKPT.exists():
        ckpt = torch.load(BEST_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        test_dice = evaluate(model, dl_test)
        print(f"[TEST] Mean Dice: {test_dice:.4f}")

    else:
        print("[WARN] No checkpoint found!")


if __name__ == "__main__":
    main()
