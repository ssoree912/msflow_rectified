# augment/cutpaste_scar.py
import math, random
import numpy as np
from PIL import Image, ImageFilter, ImageOps

def _random_affine_patch(patch, out_size):
    # random thin “scar”: affine stretch + rotate + crop to out_size
    angle = random.uniform(-25, 25)
    scale_y = random.uniform(0.05, 0.15)   # very thin height
    scale_x = random.uniform(0.6, 1.2)     # longer width
    w, h = patch.size
    nh = max(1, int(h * scale_y))
    nw = max(1, int(w * scale_x))
    patch = patch.resize((nw, nh), Image.BILINEAR)
    patch = patch.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=0)
    pw, ph = patch.size
    if pw < out_size[0] or ph < out_size[1]:
        pad_w = max(0, out_size[0]-pw); pad_h = max(0, out_size[1]-ph)
        patch = ImageOps.expand(patch, border=(0,0,pad_w,pad_h), fill=0)
    # random crop to out_size
    pw, ph = patch.size
    x = random.randint(0, pw - out_size[0]); y = random.randint(0, ph - out_size[1])
    return patch.crop((x, y, x+out_size[0], y+out_size[1]))

def cutpaste_scar(img, h_ratio=(0.01, 0.04), w_ratio=(0.15, 0.5), max_angle=30, blur_prob=0.5):
    """
    CutPaste-Scar as in the paper: paste a long-thin (scar-like) image patch from the same image,
    optionally rotated, using a mask so only the thin strip is applied.
    Returns (abnormal_image, mask) where mask is 0/255 (uint8).
    """
    assert isinstance(img, Image.Image)
    W, H = img.size

    # 1) sample target scar size (THIN!)
    scar_h = max(2, int(H * random.uniform(*h_ratio)))   # e.g., 1-4% of height
    scar_w = max(8, int(W * random.uniform(*w_ratio)))   # e.g., 15-50% of width

    # 2) sample a source region of the same size and crop it
    sx = random.randint(0, max(0, W - scar_w))
    sy = random.randint(0, max(0, H - scar_h))
    scar = img.crop((sx, sy, sx + scar_w, sy + scar_h))

    # 3) rotate thin strip & build an alpha mask for non-empty pixels
    angle = random.uniform(-max_angle, max_angle)
    scar = scar.rotate(angle, resample=Image.BILINEAR, expand=True)
    if blur_prob and random.random() < blur_prob:
        scar = scar.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # Create binary mask: any non-black pixel becomes 255 (paste region)
    # (You can make this tighter by converting to L and thresholding.)
    mask = scar.convert("L").point(lambda p: 255 if p > 0 else 0)

    # 4) sample destination and paste with mask (no rectangle artifacts)
    dx = random.randint(0, max(0, W - scar.width))
    dy = random.randint(0, max(0, H - scar.height))

    out_img = img.copy()
    out_mask = Image.new("L", (W, H), 0)
    out_img.paste(scar, (dx, dy), mask)
    out_mask.paste(mask, (dx, dy))

    return out_img, out_mask