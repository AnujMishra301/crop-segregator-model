"""
PyTorch Agricultural Dataset Loader Module
Loads drone downward-facing images and YOLO bounding box annotations.
Performs resizing to (640, 640), normalization, and tensor conversions.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class AgriculturalWeedDataset(Dataset):
    """PyTorch Dataset class for weed and crop object detection."""
    def __init__(self, image_dir, label_dir, img_size=640, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform

        if os.path.exists(image_dir):
            self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        else:
            self.image_files = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        base = os.path.splitext(fname)[0]
        img_path = os.path.join(self.image_dir, fname)
        lbl_path = os.path.join(self.label_dir, base + ".txt")

        # Open Image
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        
        # Resize image to target (img_size, img_size)
        img_resized = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)    # Channel first [3, H, W]

        # Read Labels
        boxes = []
        labels = []

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:])

                        # Convert normalized xc, yc, bw, bh to absolute [xmin, ymin, xmax, ymax] on img_size
                        xmin = (xc - bw / 2.0) * self.img_size
                        ymin = (yc - bh / 2.0) * self.img_size
                        xmax = (xc + bw / 2.0) * self.img_size
                        ymax = (yc + bh / 2.0) * self.img_size

                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(cls_id)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([orig_h, orig_w])
        }

        return img_tensor, target

def collate_fn(batch):
    """Custom collate function for object detection batching."""
    return tuple(zip(*batch))
