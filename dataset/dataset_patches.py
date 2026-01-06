# dataset_patches.py
import os, json, random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi

import torch
from torch.utils.data import Dataset, DataLoader

# ----------------------------
#  
# ----------------------------
PREPROC_IMG_DIR = Path("data_preprocessed/images")
PREPROC_MSK_DIR = Path("data_preprocessed/labels")
SPLIT_JSON      = Path("data_preprocessed/split_70_15_15.json")


TARGET_SPACING: Optional[Tuple[float,float,float]] = (1.0, 1.0, 1.0)

# ----------------------------
# 
# ----------------------------
def _is_nifti(p: Path) -> bool:
    return p.suffix == ".nii" or p.suffixes[-2:] == [".nii", ".gz"]

def _key_from_name(p: Path) -> str:
    name = p.name
    name = name.replace(".nii.gz", "").replace(".nii", "")
    parts = name.split("_")
    if len(parts) >= 2 and parts[0].lower().startswith("hippocampus"):
        return f"{parts[0].lower()}_{parts[1]}"
    return name

def collect_pairs(img_dir: Path, msk_dir: Path) -> List[Tuple[str, Path, Path]]:
    imgs = [p for p in img_dir.iterdir() if p.is_file() and _is_nifti(p)]
    msks = [p for p in msk_dir.iterdir() if p.is_file() and _is_nifti(p)]

    img_map = { _key_from_name(p): p for p in imgs }
    msk_map = { _key_from_name(p): p for p in msks }

    common = sorted(set(img_map.keys()) & set(msk_map.keys()))
    pairs = [(k, img_map[k], msk_map[k]) for k in common]
    only_img = sorted(set(img_map.keys()) - set(msk_map.keys()))
    only_msk = sorted(set(msk_map.keys()) - set(img_map.keys()))

    print(f"Found pairs: {len(pairs)}")
    if only_img: print("[WARN] Images without label (first 5):", only_img[:5])
    if only_msk: print("[WARN] Labels without image (first 5):", only_msk[:5])
    return pairs

def save_or_load_split(pairs: List[Tuple[str, Path, Path]],
                       split_json: Path,
                       train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                       seed=42):
    if split_json.exists():
        print("[INFO] Loading existing split:", split_json)
        return json.loads(split_json.read_text(encoding="utf-8"))

    keys = [k for k, _, _ in pairs]
    rng = random.Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train_keys = keys[:n_train]
    val_keys   = keys[n_train:n_train+n_val]
    test_keys  = keys[n_train+n_val:]

    split = {"train": train_keys, "val": val_keys, "test": test_keys}
    split_json.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print("[INFO] Split saved:", split_json)
    return split

# ----------------------------
# Resample و Affine
# ----------------------------
def voxel_sizes_from_affine(affine: np.ndarray) -> np.ndarray:

    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

def affine_with_new_spacing(affine: np.ndarray, new_spacing: Tuple[float, float, float]) -> np.ndarray:

    out = affine.copy()
    R = affine[:3, :3]

    dirs = R / np.linalg.norm(R, axis=0, keepdims=True)
    out[:3, :3] = dirs * np.asarray(new_spacing)[None, :]
    return out

def resample_array_to_spacing(
    arr: np.ndarray,
    affine: np.ndarray,
    target_spacing: Tuple[float, float, float],
    order: int
) -> Tuple[np.ndarray, np.ndarray]:

    current_spacing = voxel_sizes_from_affine(affine)  # (3,)
    zoom_factors = current_spacing / np.asarray(target_spacing, dtype=float)

    arr_rs = ndi.zoom(arr, zoom=zoom_factors, order=order, mode="nearest")
    new_affine = affine_with_new_spacing(affine, target_spacing)
    return arr_rs.astype(arr.dtype, copy=False), new_affine

def load_and_resample_nifti(
    path: Path,
    target_spacing: Optional[Tuple[float, float, float]] = None,
    is_mask: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
   
    nii = nib.load(str(path))
    nii = nib.as_closest_canonical(nii)         
    data = nii.get_fdata(dtype=np.float32)    
    if data.ndim == 4:
        data = data[..., 0]
    affine = nii.affine

    if target_spacing is not None:
        order = 0 if is_mask else 3
        data, affine = resample_array_to_spacing(data, affine, target_spacing, order=order)

    if is_mask:
        data = (data > 0.5).astype(np.uint8)

    return data, affine

# ----------------------------
# Dataset: Full-volume // Patch-based 
# ----------------------------
class HippocampusDataset(Dataset):
    def __init__(self,
                 pairs: List[Tuple[str, Path, Path]],
                 split_keys: List[str],
                 mode: str = "patch",
                 patch_size: Tuple[int,int,int]=(96,96,96),
                 fg_ratio: float = 0.7,           
                 max_tries: int = 10,
                 target_spacing: Optional[Tuple[float,float,float]] = TARGET_SPACING):
        
        self.items = [t for t in pairs if t[0] in split_keys]
        self.mode = mode
        self.patch_size = np.array(patch_size, dtype=int)
        self.fg_ratio = float(fg_ratio)
        self.max_tries = int(max_tries)
        self.target_spacing = target_spacing

        assert len(self.items) > 0, "Selected split is empty!"

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _to_tensor(img: np.ndarray, msk: np.ndarray):
        # (D,H,W) → (C,D,H,W) C=1
        img_t = torch.from_numpy(img.astype(np.float32))[None, ...]
        msk_t = torch.from_numpy(msk.astype(np.uint8))[None, ...]
        return img_t, msk_t

    @staticmethod
    def _valid_start(end_lim, size):
     
        return 0 if end_lim <= size else np.random.randint(0, end_lim - size + 1)

    def _sample_patch(self, I: np.ndarray, M: np.ndarray):
        D,H,W = I.shape
        pd,ph,pw = self.patch_size

      
        need_pad = np.maximum(self.patch_size - np.array([D,H,W]), 0)
        if np.any(need_pad > 0):
            pad_b = need_pad // 2
            pad_a = need_pad - pad_b
            padw = [(int(pad_b[0]), int(pad_a[0])),
                    (int(pad_b[1]), int(pad_a[1])),
                    (int(pad_b[2]), int(pad_a[2]))]
            I = np.pad(I, padw, mode="constant", constant_values=0)
            M = np.pad(M, padw, mode="constant", constant_values=0)
            D,H,W = I.shape

     
        choose_fg = (np.random.rand() < self.fg_ratio) and (M.sum() > 0)
        center = None
        if choose_fg:
            fg_idx = np.argwhere(M > 0)
            for _ in range(self.max_tries):
                z,y,x = fg_idx[np.random.randint(0, fg_idx.shape[0])]
                z0 = np.clip(z - pd//2, 0, max(0, D - pd))
                y0 = np.clip(y - ph//2, 0, max(0, H - ph))
                x0 = np.clip(x - pw//2, 0, max(0, W - pw))
                z1, y1, x1 = z0+pd, y0+ph, x0+pw
                if z1 <= D and y1 <= H and x1 <= W:
                    center = (z0, y0, x0)
                    break

        if center is None:
            z0 = self._valid_start(D, pd)
            y0 = self._valid_start(H, ph)
            x0 = self._valid_start(W, pw)
        else:
            z0, y0, x0 = center

        z1, y1, x1 = z0+pd, y0+ph, x0+pw
        I_patch = I[z0:z1, y0:y1, x0:x1]
        M_patch = M[z0:z1, y0:y1, x0:x1]
        return I_patch, M_patch

    def __getitem__(self, idx):
        key, img_p, msk_p = self.items[idx]

       
        I, aff_I = load_and_resample_nifti(img_p, target_spacing=self.target_spacing, is_mask=False)
        M, aff_M = load_and_resample_nifti(msk_p, target_spacing=self.target_spacing, is_mask=True)

        if I.shape != M.shape:
            scale = np.array(I.shape) / np.array(M.shape)
            M = ndi.zoom(M, zoom=scale, order=0) 

        if self.mode == "full":
            img_t, msk_t = self._to_tensor(I, M)  # (1,D,H,W)
            sample = {"key": key, "image": img_t, "mask": msk_t, "affine": aff_I}
        else:  # patch mode
            Ip, Mp = self._sample_patch(I, M)
            img_t, msk_t = self._to_tensor(Ip, Mp)
            sample = {"key": key, "image": img_t, "mask": msk_t, "affine": aff_I}

        return sample

# ----------------------------
# 
# ----------------------------
if __name__ == "__main__":
 
    pairs = collect_pairs(PREPROC_IMG_DIR, PREPROC_MSK_DIR)

 
    split = save_or_load_split(pairs, SPLIT_JSON, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

   
    ds_train = HippocampusDataset(
        pairs, split["train"],
        mode="patch",
        patch_size=(96,96,96),
        fg_ratio=0.7,
        target_spacing=TARGET_SPACING,
    )
    ds_val = HippocampusDataset(
        pairs, split["val"],
        mode="patch",
        patch_size=(96,96,96),
        fg_ratio=0.7,
        target_spacing=TARGET_SPACING,
    )
    ds_test = HippocampusDataset(
        pairs, split["test"],
        mode="full",
        target_spacing=TARGET_SPACING,
    )

    # 4) DataLoader
    dl_train = DataLoader(ds_train, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    
    b = next(iter(dl_train))
    print("Train batch -> image:", tuple(b["image"].shape), "mask:", tuple(b["mask"].shape), "key:", b["key"][0])
    print("mask unique:", torch.unique(b["mask"]))
    print("voxel sizes (approx.):", voxel_sizes_from_affine(b["affine"].numpy()))
