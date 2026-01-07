import os
import re
import bisect
import json
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

__all__ = ('MVTecDataset', 'VisADataset', 'MSTCDataset', 'MSTCFeatureDataset')

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



MSTC_CLASS_NAMES = ['shanghaitech']  # single class placeholder

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
        self.pixel_eval = getattr(c, 'pixel_eval', True)

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
            m = re.search(r'(\d+)$', stem)
            fidx = int(m.group(1)) if m else 0

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
                if self.pixel_eval and os.path.isfile(pm_path):
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

        if not self.pixel_eval:
            mask = torch.zeros([1, *self.input_size], dtype=torch.float32)
            return img, y, mask

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


class MSTCFeatureDataset(Dataset):
    """
    Load precomputed per-video features saved as <video_id>_res.npy.
    Each file contains a dict with keys: s1, s2, s3, labels, frame_indices, paths.
    Returns (s1, s2, s3, y, mask), where mask is zeros (pixel eval disabled).
    """
    def __init__(self, c, is_train=True):
        self.c = c
        self.is_train = is_train
        split_dir = 'training' if is_train else 'testing'
        root = getattr(c, 'feature_cache_dir', None) or c.data_path
        subdir = getattr(c, 'feature_subdir', 'features')
        suffix = getattr(c, 'feature_suffix', '_res.npy')
        self.feature_dir = os.path.join(root, split_dir, subdir)
        if not os.path.isdir(self.feature_dir):
            raise FileNotFoundError(f'feature_dir not found: {self.feature_dir}')
        self.files = sorted([
            os.path.join(self.feature_dir, f)
            for f in os.listdir(self.feature_dir)
            if f.endswith(suffix)
        ])
        if not self.files:
            raise FileNotFoundError(f'no feature files found under: {self.feature_dir}')

        self._counts = []
        self._cache_idx = None
        self._cache_data = None
        self._suffix = suffix
        manifest_path = os.path.join(self.feature_dir, 'manifest.json')
        video_counts = {}
        stage_channels = None
        stage_shapes = None
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                video_counts = manifest.get('video_counts', {}) or {}
                stage_channels = manifest.get('stage_channels')
                stage_shapes = manifest.get('stage_shapes')
            except Exception:
                video_counts = {}

        if stage_channels and stage_shapes:
            self.output_channels = [int(x) for x in stage_channels]
            self.stage_hw = [(int(h), int(w)) for h, w in stage_shapes]
        else:
            data = np.load(self.files[0], allow_pickle=True).item()
            self._cache_idx = 0
            self._cache_data = data
            s1, s2, s3 = data["s1"], data["s2"], data["s3"]
            self.output_channels = [s1.shape[1], s2.shape[1], s3.shape[1]]
            self.stage_hw = [(s1.shape[2], s1.shape[3]),
                             (s2.shape[2], s2.shape[3]),
                             (s3.shape[2], s3.shape[3])]

        missing_counts = 0
        for fpath in self.files:
            vid = os.path.basename(fpath)
            if vid.endswith(suffix):
                vid = vid[:-len(suffix)]
            if vid not in video_counts:
                missing_counts += 1
        if missing_counts:
            print(f"[FeatureCache] indexing {missing_counts} feature files (this may take a while)")

        scanned = 0
        for i, fpath in enumerate(self.files):
            vid = os.path.basename(fpath)
            if vid.endswith(suffix):
                vid = vid[:-len(suffix)]
            count = video_counts.get(vid)
            if count is None:
                if i == 0 and self._cache_data is not None:
                    data = self._cache_data
                else:
                    data = np.load(fpath, allow_pickle=True).item()
                count = int(data["s1"].shape[0])
                scanned += 1
                if scanned % 50 == 0:
                    print(f"[FeatureCache] indexed {scanned}/{missing_counts} files")
            self._counts.append(int(count))
        self._cum_counts = np.cumsum(self._counts)
        self._feature_fp32 = getattr(c, "feature_fp32", True)

    def __len__(self):
        return int(self._cum_counts[-1])

    def _load_file(self, file_idx):
        if self._cache_idx == file_idx and self._cache_data is not None:
            return self._cache_data
        data = np.load(self.files[file_idx], allow_pickle=True).item()
        self._cache_idx = file_idx
        self._cache_data = data
        return data

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self._cum_counts, idx)
        start = 0 if file_idx == 0 else self._cum_counts[file_idx - 1]
        local_idx = idx - start
        data = self._load_file(file_idx)

        s1 = torch.from_numpy(data["s1"][local_idx])
        s2 = torch.from_numpy(data["s2"][local_idx])
        s3 = torch.from_numpy(data["s3"][local_idx])
        if self._feature_fp32:
            s1 = s1.float()
            s2 = s2.float()
            s3 = s3.float()
        label = int(data["labels"][local_idx])
        mask = torch.zeros([1, *self.c.input_size], dtype=torch.float32)
        return s1, s2, s3, label, mask
