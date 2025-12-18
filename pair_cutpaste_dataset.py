# pair_cutpaste_dataset.py

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from datasets import MVTecDataset, VisADataset
from augment.cutpaste_scar import cutpaste_scar

class PairCutPasteMVTec(Dataset):
    """
    Returns a pair (x_clean, x_abn, y, mask) for training velocity nets on MVTec.
    Uses only TRAIN split (normal images). y=0 and mask=zeros.
    """
    def __init__(self, c, class_name: str):
        self.c = c
        self.ds = MVTecDataset(c, is_train=True)
        # filter non-files (e.g., .ipynb_checkpoints)
        self.paths = [p for p in getattr(self.ds, "x", []) if os.path.isfile(p)]

        # Reuse dataset's normalization if available, else default to ImageNet stats
        self.normalize = getattr(self.ds, "normalize", T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ))
        self.resize_pil = T.Resize(c.input_size, InterpolationMode.LANCZOS)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x_path = self.paths[idx]
        x_pil = Image.open(x_path)

        # MVTec some classes are grayscale; ensure RGB
        if x_pil.mode != 'RGB':
            x_pil = x_pil.convert('RGB')

        # Resize before CutPaste so scar params match network input size
        x_pil = self.resize_pil(x_pil)

        # Abnormal via CutPaste-Scar (PIL → PIL)
        x_abn_pil, _ = cutpaste_scar(x_pil)

        # Tensor + normalize
        x_clean = self.normalize(self.to_tensor(x_pil))
        x_abn   = self.normalize(self.to_tensor(x_abn_pil))

        y = 0  # training on normal
        mask = torch.zeros([1, *self.c.input_size], dtype=torch.float32)

        return x_clean, x_abn, y, mask


class PairCutPasteVisA(Dataset):
    """
    Returns a pair (x_clean, x_abn, y, mask) for training velocity nets on VisA.
    Uses only TRAIN split (normal images). y=0 and mask=zeros.
    """
    def __init__(self, c, class_name: str):
        self.c = c
        self.ds = VisADataset(c, is_train=True)
        # robust path filtering
        self.paths = [p for p in getattr(self.ds, "x", []) if os.path.isfile(p)]

        # Reuse dataset normalization if provided; else default to ImageNet stats
        self.normalize = getattr(self.ds, "normalize", T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        ))
        self.resize_pil = T.Resize(c.input_size, InterpolationMode.LANCZOS)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x_path = self.paths[idx]
        x_pil = Image.open(x_path).convert('RGB')  # VisA images are RGB; enforce

        x_pil = self.resize_pil(x_pil)
        x_abn_pil, _ = cutpaste_scar(x_pil)

        x_clean = self.normalize(self.to_tensor(x_pil))
        x_abn   = self.normalize(self.to_tensor(x_abn_pil))

        y = 0
        mask = torch.zeros([1, *self.c.input_size], dtype=torch.float32)

        return x_clean, x_abn, y, mask
