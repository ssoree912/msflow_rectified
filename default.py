import torch
import os

# base
seed = 9826
gpu = '1'
device = torch.device("cuda")
mode = 'train'
eval_ckpt = ''
resume = False

# optimizer
meta_epochs = 25 # totally 100
sub_epochs = 4
lr = 1e-4
lr_decay_milestones = [70, 90]
lr_decay_gamma = 0.33
lr_warmup = True
lr_warmup_from = 0.1
lr_warmup_epochs = 3
batch_size = 16
workers = 4


# dataset
dataset = 'mvtec' # [mvtec, visa]
class_name = 'zipper'
input_size = (512, 512)
img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
mstc_frame_stride = 5

# model
extractor = 'wide_resnet50_2' # [resnet18, resnet34, resnet50, resnext50_32x4d, wide_resnet50_2]
pool_type = 'avg'
c_conds = [64, 64, 64]
parallel_blocks = [2, 5, 8]
clamp_alpha = 1.9

# pruning (extractor)
pruning_mode = 'dense'
pruning_sparsity = 0.0
dwa_alpha = 1.0
dwa_beta = 1.0
dwa_update_threshold = False
dwa_threshold_percentile = 50

# evaluation
top_k = 0.03
pro_eval = True
pro_eval_interval = 6
pixel_eval = True
pixel_eval_downscale = 1

# feature cache (mstc)
feature_cache = False
feature_cache_dir = None
feature_subdir = 'features'
feature_suffix = '_res.npy'
feature_fp32 = True

# results
work_dir = './work_dirs'

# logging
log_interval = 50
