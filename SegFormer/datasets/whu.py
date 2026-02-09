import os
import json
import warnings
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from pycocotools import mask as maskUtils
    _HAS_PYCOCO = True
except Exception:
    maskUtils = None
    _HAS_PYCOCO = False


class WhuBuildingDataset(Dataset):
    """
    WHU Building Rooftop dataset loader (COCO-style JSON -> binary mask).

    Expected folder structure:
      whu-building-rooftop-dataset/
        annotation/
          train.json
          validation.json
          test.json
        train/
          100000.TIF
          ...
        val/
          20000.TIF

    Usage:
      ds = WhuBuildingDataset(root_dir, split='train', transform=albumentations_transform)
      img, mask = ds[0]

    Notes:
      - Handles polygon segmentations (lists of x,y floats in COCO format).
      - For RLE segmentations (dict form) `pycocotools` is required.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[object] = None,
        img_exts: Tuple[str, ...] = (".TIF", ".tif", ".jpg", ".png"),
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.img_exts = img_exts
        self.ignore_label = 255  # For compatibility with training script (binary seg uses 0/1)

        # Determine annotation filename
        ann_name = "validation.json" if split in ("val", "validation") else f"{split}.json"
        self.ann_path = os.path.join(root_dir, "annotation", ann_name)

        if not os.path.exists(self.ann_path):
            raise FileNotFoundError(f"Annotation file not found: {self.ann_path}")

        self.img_dir = os.path.join(root_dir, split)
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # Load COCO-style JSON and build mapping image_name -> list[segmentations]
        with open(self.ann_path, "r") as f:
            data = json.load(f)

        # Build image id -> file_name map
        self._id2name = {img["id"]: img["file_name"] for img in data.get("images", [])}

        # Map filename -> list of segmentations
        self._ann_map = {}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            fname = self._id2name.get(img_id)
            if fname is None:
                continue
            seg = ann.get("segmentation")
            if seg is None:
                continue
            self._ann_map.setdefault(fname, []).append(seg)

        # Build list of images present in folder and in annotations (preserve folder order)
        available = set(os.listdir(self.img_dir))
        imgs = [f for f in available if f.lower().endswith(self.img_exts)]

        # Prefer images that exist in annotation list, but include all images in folder
        annotated = [f for f in imgs if f in self._ann_map]
        unannotated = [f for f in imgs if f not in self._ann_map]
        self.images = annotated + unannotated

    def __len__(self):
        return len(self.images)

    def _create_mask_from_segs(self, segs: List, h: int, w: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)

        for seg in segs:
            # Polygon format (list of polygons or single polygon)
            if isinstance(seg, list):
                # COCO allows either a list of polygons (each polygon is a flat list)
                # or a single polygon (flat list of floats). Handle both.
                if len(seg) == 0:
                    continue
                # If first element is a list -> multiple polygons
                if isinstance(seg[0], list):
                    for poly in seg:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                        pts = np.round(pts).astype(np.int32)
                        if pts.shape[0] >= 3:
                            cv2.fillPoly(mask, [pts], 1)
                else:
                    pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    pts = np.round(pts).astype(np.int32)
                    if pts.shape[0] >= 3:
                        cv2.fillPoly(mask, [pts], 1)

            # RLE or compressed RLE (requires pycocotools)
            elif isinstance(seg, dict):
                if not _HAS_PYCOCO:
                    warnings.warn(
                        "Encountered RLE segmentation but `pycocotools` is not installed; skipping RLE mask."
                    )
                    continue
                try:
                    m = maskUtils.decode(seg)
                    # 'm' could have shape (H, W) or (H, W, 1)
                    if m.ndim == 3:
                        m = m[:, :, 0]
                    mask = np.maximum(mask, m.astype(np.uint8))
                except Exception:
                    warnings.warn("Failed to decode RLE segmentation; skipping this annotation.")
                    continue

        return mask

    def __getitem__(self, index: int):
        img_name = self.images[index]
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            # return a black image and empty mask if reading fails
            # choose a sane default size (512x512)
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        segs = self._ann_map.get(img_name, [])
        mask = self._create_mask_from_segs(segs, h, w)

        # Convert mask to float32 0/1
        mask = mask.astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        return image, mask


def get_whu_loaders(
    root_dir: str,
    train_split: str = "train",
    val_split: str = "val",
    transform_train: Optional[object] = None,
    transform_val: Optional[object] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    train_ds = WhuBuildingDataset(root_dir=root_dir, split=train_split, transform=transform_train)
    val_ds = WhuBuildingDataset(root_dir=root_dir, split=val_split, transform=transform_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Minimal example (requires albumentations if you want transforms)
    from albumentations import Compose, Normalize
    from albumentations.pytorch import ToTensorV2

    transform = Compose([Normalize(), ToTensorV2()])
    root = os.path.join("..", "whu-building-rooftop-dataset")
    try:
        tr, va = get_whu_loaders(root, transform_train=transform, transform_val=transform, batch_size=2)
        x, y = next(iter(tr))
        print("Batch shapes:", x.shape, y.shape)
    except Exception as e:
        print("Example run failed:", e)
