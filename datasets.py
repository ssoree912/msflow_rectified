import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

__all__ = ('MVTecDataset', 'VisADataset', 'MSTCDataset')

MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

class MVTecDataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in MVTEC_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.input_size = c.input_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        #x = Image.open(x).convert('RGB')
        x = Image.open(x)
        if self.class_name in ['zipper', 'screw', 'grid']:  # handle greyscale classes
            x = np.expand_dims(np.array(x), axis=2)
            x = np.concatenate([x, x, x], axis=2)
            
            x = Image.fromarray(x.astype('uint8')).convert('RGB')
        #
        x = self.normalize(self.transform_x(x))
        #
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)

VISA_CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 
                    'fryum', 'macaroni1', 'macaroni2', 
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

class VisADataset(Dataset):
    def __init__(self, c, is_train=True):
        assert c.class_name in VISA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, MVTEC_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.input_size = c.input_size
        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.input_size, InterpolationMode.LANCZOS),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.input_size, InterpolationMode.NEAREST),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.img_mean, c.img_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        x = Image.open(x)
        x = self.normalize(self.transform_x(x))
        if y == 0:
            mask = torch.zeros([1, *self.input_size])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)



MSTC_CLASS_NAMES = ['campus']  # single “class” placeholder

class MSTCDataset(Dataset):
    """
    mSTC (mini ShanghaiTech Campus) frame dataset compatible with MSFlow.
    Directory layout (you said):
      data/shanghaitech/
        training/
          frames/            # folders like 01_0014/ with 0.jpg, 1.jpg, ...
        testing/
          frames/            # folders like 01_0014/ with 000.jpg, 001.jpg, ...
          test_frame_mask/   # .npy per video: e.g., 01_0014.npy with shape [N_frames]
          test_pixel_mask/   # .npy per video: e.g., 01_0014.npy with shape [N_frames, H, W]
    Returns (x, y, mask) where:
      - x: normalized RGB tensor resized to c.input_size (256, 384)
      - y: frame-level abnormal label (0/1)
      - mask: pixel mask [1,H,W] (zeros for train/normal frames)
    """
    def __init__(self, c, is_train=True):
        assert c.class_name in MSTC_CLASS_NAMES, f'class_name: {c.class_name}, should be one of {MSTC_CLASS_NAMES}'
        self.c = c
        self.is_train = is_train
        self.root = c.data_path  # e.g., ./data/shanghaitech
        self.input_size = c.input_size  # (256,384)

        split_dir = 'training' if is_train else 'testing'
        self.frames_dir = os.path.join(self.root, split_dir, 'frames')

        # gather all frame paths: frames/<vid_id>/*.jpg
        vid_ids = [d for d in sorted(os.listdir(self.frames_dir)) if os.path.isdir(os.path.join(self.frames_dir, d))]
        frame_paths = []
        for vid in vid_ids:
            vid_dir = os.path.join(self.frames_dir, vid)
            # jpgs may be 0.jpg.. or 000.jpg.. — sort lexicographically works with zero-padding
            for fname in sorted(os.listdir(vid_dir)):
                p = os.path.join(vid_dir, fname)
                if os.path.isfile(p) and os.path.splitext(p)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                    frame_paths.append(p)

        self.x = frame_paths
        self.y = [0] * len(self.x)   # filled for test below
        self.mask = [None] * len(self.x)

        # build index to video/frame
        # path .../frames/<vid>/<frame>.jpg
        self._vid_of = []
        self._idx_in_vid = []
        self._vid_start_index = {}  # to speed label/mask lookup

        # lazily load masks for test
        self._frame_labels = {}   # vid -> np.ndarray shape [N]
        self._pixel_masks = {}    # vid -> np.ndarray shape [N,H,W] or [N,h,w]

        # scan and fill indices
        last_vid = None
        count = 0
        for i, p in enumerate(self.x):
            vid = os.path.basename(os.path.dirname(p))
            stem = os.path.splitext(os.path.basename(p))[0]
            try:
                fidx = int(stem)  # 0-based or 000-based
            except:
                # fallback: strip leading zeros
                fidx = int(stem.lstrip('0') or '0')

            self._vid_of.append(vid)
            self._idx_in_vid.append(fidx)
            if vid != last_vid:
                self._vid_start_index[vid] = count
                last_vid = vid
            count += 1

        if not is_train:
            # load test masks (frame-level & pixel-level) per video
            frame_mask_dir = os.path.join(self.root, 'testing', 'test_frame_mask')
            pixel_mask_dir = os.path.join(self.root, 'testing', 'test_pixel_mask')
            for vid in vid_ids:
                fm_path = os.path.join(frame_mask_dir, f'{vid}.npy')
                pm_path = os.path.join(pixel_mask_dir, f'{vid}.npy')
                if os.path.isfile(fm_path):
                    self._frame_labels[vid] = np.load(fm_path)  # shape [N]
                else:
                    self._frame_labels[vid] = None
                if os.path.isfile(pm_path):
                    self._pixel_masks[vid] = np.load(pm_path)   # shape [N,H,W] (bool/int)
                else:
                    self._pixel_masks[vid] = None

        # transforms
        self.transform_x = T.Compose([
            T.Resize(self.input_size, InterpolationMode.LANCZOS),
            T.ToTensor()
        ])
        self.transform_mask = T.Compose([
            T.Resize(self.input_size, InterpolationMode.NEAREST),
            T.ToTensor()
        ])
        self.normalize = T.Normalize(c.img_mean, c.img_std)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        p = self.x[idx]
        img = Image.open(p).convert('RGB')
        img = self.normalize(self.transform_x(img))

        if self.is_train:
            # mSTC train is normal-only
            y = 0
            mask = torch.zeros([1, *self.input_size], dtype=torch.float32)
            return img, y, mask

        # test:
        vid = self._vid_of[idx]
        fidx = self._idx_in_vid[idx]

        # frame label
        y = 0
        if self._frame_labels.get(vid) is not None and fidx < len(self._frame_labels[vid]):
            y = int(self._frame_labels[vid][fidx])

        # pixel mask (if provided)
        if self._pixel_masks.get(vid) is not None and fidx < len(self._pixel_masks[vid]):
            m = self._pixel_masks[vid][fidx]
            # ensure [H,W] binary
            m = (m > 0).astype(np.uint8)
            m = Image.fromarray(m * 255)
            mask = self.transform_mask(m)  # -> [1,H,W] in {0,1}
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros([1, *self.input_size], dtype=torch.float32)

        return img, y, mask