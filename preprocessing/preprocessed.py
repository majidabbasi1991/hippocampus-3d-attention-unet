"""
Offline MRI Preprocessing Pipeline (All-in-one)
Dataset:
- Medical Segmentation Decathlon (Task04 Hippocampus)
- No clinical or private patient data used

pipeline steps:
- format filter (.nii / .nii.gz)
- name normalization & best-unique selection
- pairing image/label
- canonical, resample, new affine
- center crop/pad to fixed shape
- z-score (ignore zeros)
- binarize mask after resample
- save with fresh headers
- metadata (JSON) for reproducibility
"""

from __future__ import annotations
import re, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# ============== 0) Utils: File recognition & normalization ==============
def is_nifti(p: Path) -> bool:
   
    return p.suffix == ".nii" or p.suffixes[-2:] == [".nii", ".gz"]

def normalize_name(p: Path) -> Optional[str]:
    """
      hippocampus_001.nii.gz      -> hippocampus_001
      hippocampus_001 (3).nii     -> hippocampus_001
      ._hippocampus_001.nii.gz    -> None  
    """
    name = p.name
    if name.startswith("._"):
        return None
    name = re.sub(r"\s*\(\d+\)", "", name) 
    name = re.sub(r"\.nii(\.gz)?$", "", name, flags=re.IGNORECASE)
    m = re.search(r"(hippocampus)_(\d+)", name, flags=re.IGNORECASE)  
    if not m:
        return None
    prefix = m.group(1).lower()
    num = int(m.group(2))
    return f"{prefix}_{num:03d}"

def collect_best_unique(files_dir: Path) -> Dict[str, Path]:
    
    cand: Dict[str, Path] = {}
    for p in files_dir.iterdir():
        if not (p.is_file() and is_nifti(p)):
            continue
        key = normalize_name(p)
        if not key:
            continue
        if key not in cand:
            cand[key] = p
        else:
            cur = cand[key]
            if p.stat().st_size > cur.stat().st_size:
                cand[key] = p
            elif (cur.suffix != ".gz") and (p.suffix == ".gz"):
                cand[key] = p
    return cand

def safe_stem(p: Path) -> str:
    n = p.name
    if n.lower().endswith(".nii.gz"): return n[:-7]
    if n.lower().endswith(".nii"):    return n[:-4]
    return p.stem

# ============== 1) Geometry & intensity helpers ==============
def load_nii(path: Path):
    nii = nib.load(str(path))
    nii = nib.as_closest_canonical(nii)  
    arr = nii.get_fdata(dtype=np.float32)
    if arr.ndim == 4:  
        arr = arr[..., 0]
    A = nii.affine
    vx = nib.affines.voxel_sizes(A)[:3]
    return nii, arr, A, vx

def resample_array(arr: np.ndarray, old_vx, new_vx, is_label=False) -> np.ndarray:
    factors = np.array(old_vx, float) / np.array(new_vx, float)
    order = 0 if is_label else 3
    return zoom(arr, zoom=factors, order=order)

def update_affine_after_resample(A_old, old_vx, new_vx):
    A = np.array(A_old, float).copy()
    Rdir = A[:3, :3] @ np.diag(1.0 / np.array(old_vx, float))
    A[:3, :3] = Rdir @ np.diag(np.array(new_vx, float))
    return A

def center_crop_pad(arr: np.ndarray, target_shape: Tuple[int,int,int]):
    cur = np.array(arr.shape[:3], int)
    tgt = np.array(target_shape, int)
    start = np.maximum((cur - tgt)//2, 0)
    end   = np.minimum(start + tgt, cur)
    slc = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
    cropped = arr[slc]
    new_cur = np.array(cropped.shape[:3], int)
    need_pad = np.maximum(tgt - new_cur, 0)
    pad_b = need_pad // 2
    pad_a = need_pad - pad_b
    pad_w = [(int(b), int(a)) for b, a in zip(pad_b, pad_a)]
    if cropped.ndim > 3:
        pad_w += [(0,0)]*(cropped.ndim-3)
    out = np.pad(cropped, pad_w, mode="constant", constant_values=0)
    return out, start, pad_b

def adjust_affine_for_crop_pad(A_resampled, start_idx, pad_before):
    A = np.array(A_resampled, float).copy()
    shift = np.array(start_idx, float) - np.array(pad_before, float)
    A[:3, 3] = A[:3, 3] + A[:3, :3] @ shift
    return A

def zscore_intensity(arr: np.ndarray) -> np.ndarray:
    mask = arr != 0
    if not np.any(mask):
        return arr
    m = arr[mask].mean()
    s = arr[mask].std()
    if s < 1e-6:
        return arr
    out = arr.copy()
    out[mask] = (out[mask] - m) / s
    return out

# ============== 2) Preprocessor Class (All-in-one) ==============
class MRIPreprocessor:
    """
    
      pre = MRIPreprocessor(
          images_dir=".../imagesTr",
          labels_dir=".../labelsTr",
          out_img_dir=".../preproc/images",
          out_msk_dir=".../preproc/labels",
          target_vx=(1,1,1),
          target_shape=(128,128,128),
          zscore=True,
          mask_thresh=0.5,
          metadata_path=".../preproc/preproc_meta.json"
      )
      pre.run()

   
    """
    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        out_img_dir: str | Path,
        out_msk_dir: str | Path,
        target_vx=(1.0, 1.0, 1.0),
        target_shape: Optional[Tuple[int,int,int]] = (128, 128, 128),
        zscore: bool = True,
        mask_thresh: float = 0.5,
        metadata_path: Optional[str | Path] = None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.out_img_dir = Path(out_img_dir)
        self.out_msk_dir = Path(out_msk_dir)

        
        self.out_img_dir.mkdir(parents=True, exist_ok=True)
        self.out_msk_dir.mkdir(parents=True, exist_ok=True)

        self.target_vx = np.array(target_vx, np.float32)
        self.target_shape = target_shape
        self.zscore = bool(zscore)
        self.mask_thresh = float(mask_thresh)

        
        self.metadata_path = Path(metadata_path) if metadata_path else (self.out_img_dir / "preproc_meta.json")

    # ---------- pairing ----------
    def _pair_image_label(self) -> List[Tuple[str, Path, Path]]:
       
        img_map = collect_best_unique(self.images_dir)
        msk_map = collect_best_unique(self.labels_dir)

        common = sorted(set(img_map.keys()) & set(msk_map.keys()))
        only_img = sorted(set(img_map.keys()) - set(msk_map.keys()))
        only_msk = sorted(set(msk_map.keys()) - set(img_map.keys()))

        print(f"Pairs found: {len(common)}")
        if only_img:
            print("[Missing LABEL for]:", only_img[:10], "..." if len(only_img) > 10 else "")
        if only_msk:
            print("[Missing IMAGE for]:", only_msk[:10], "..." if len(only_msk) > 10 else "")

        return [(k, img_map[k], msk_map[k]) for k in common]

    # ---------- one-case process ----------
    def _process_one(self, key: str, img_p: Path, msk_p: Path):
        # load
        _, I, A_img, vx_img = load_nii(img_p)
        _, M, A_msk, vx_msk = load_nii(msk_p)

        if not np.allclose(vx_img, vx_msk, atol=1e-5):
            print(f"[WARN voxel mismatch] {key}: img={vx_img} vs lab={vx_msk}")

        # resample (image cubic, mask nearest + binarize)
        I_rs = resample_array(I, vx_img, self.target_vx, is_label=False)
        A_img_rs = update_affine_after_resample(A_img, vx_img, self.target_vx)

        M_rs = resample_array(M, vx_msk, self.target_vx, is_label=True)
        M_rs = (M_rs > self.mask_thresh).astype(np.uint8)
        A_msk_rs = update_affine_after_resample(A_msk, vx_msk, self.target_vx)

        # shape std
        if self.target_shape is not None:
            I_std, start, pad_b = center_crop_pad(I_rs, self.target_shape)
            A_img_std = adjust_affine_for_crop_pad(A_img_rs, start, pad_b)

            s = start
            e = s + np.minimum(np.array(self.target_shape), np.array(M_rs.shape[:3]))
            slc = tuple(slice(int(s[i]), int(e[i])) for i in range(3))
            M_crop = M_rs[slc]
            need_pad = np.maximum(np.array(self.target_shape) - np.array(M_crop.shape[:3]), 0)
            pad_a = need_pad - pad_b
            pad_w = [(int(b), int(a)) for b, a in zip(pad_b, pad_a)]
            M_std = np.pad(M_crop, pad_w, mode="constant", constant_values=0)
            A_msk_std = adjust_affine_for_crop_pad(A_msk_rs, start, pad_b)
        else:
            I_std, A_img_std = I_rs, A_img_rs
            M_std, A_msk_std = M_rs, A_msk_rs

        # intensity
        if self.zscore:
            I_std = zscore_intensity(I_std)

        # save (fresh headers)
        hdr_img = nib.Nifti1Header(); hdr_img.set_data_dtype(np.float32)
        hdr_msk = nib.Nifti1Header(); hdr_msk.set_data_dtype(np.uint8)

        out_img = self.out_img_dir / f"{key}_iso{int(self.target_vx[0])}mm_std.nii.gz"
        out_msk = self.out_msk_dir / f"{key}_iso{int(self.target_vx[0])}mm_std.nii.gz"

        nib.save(nib.Nifti1Image(I_std.astype(np.float32), A_img_std, header=hdr_img), str(out_img))
        nib.save(nib.Nifti1Image(M_std.astype(np.uint8),  A_msk_std, header=hdr_msk), str(out_msk))

        return out_img, out_msk

    # ---------- metadata ----------
    def _write_metadata(self, total_pairs: int):
        
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "images_dir": str(self.images_dir),
            "labels_dir": str(self.labels_dir),
            "out_img_dir": str(self.out_img_dir),
            "out_msk_dir": str(self.out_msk_dir),
            "target_vx": self.target_vx.tolist(),
            "target_shape": list(self.target_shape) if self.target_shape else None,
            "zscore": self.zscore,
            "mask_thresh": self.mask_thresh,
            "pairs": total_pairs,
            "version": 1
        }
        self.metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---------- run ----------
    def run(self):
        pairs = self._pair_image_label()
        self._write_metadata(len(pairs))
        saved = 0
        for key, img_p, msk_p in pairs:
            try:
                self._process_one(key, img_p, msk_p)
                saved += 1
                if saved % 10 == 0:
                    print(f"[OK] {saved}/{len(pairs)} saved... Last: {key}")
            except Exception as e:
                print(f"[ERR] {key} -> {e}")
        print(f"Done. Saved: {saved}/{len(pairs)}")


# ============================
if __name__ == "__main__":
    pre = MRIPreprocessor(
        images_dir="data/Task04_Hippocampus/imagesTr",
        labels_dir="data/Task04_Hippocampus/labelsTr",
        out_img_dir="data_preprocessed/images",
        out_msk_dir="data_preprocessed/labels",
        target_vx=(1,1,1),
        target_shape=(128,128,128),
        zscore=True,
        mask_thresh=0.5,
        metadata_path="data_preprocessed/preproc_meta.json"
    )
    pre.run()
