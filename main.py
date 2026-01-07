import os, random
import numpy as np
import torch
import argparse
import wandb

from train import train

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parsing_args(c):
    parser = argparse.ArgumentParser(description='msflow')
    parser.add_argument('--dataset', default='mvtec', type=str, 
                        choices=['mvtec', 'visa', 'mstc'], help='dataset name')
    parser.add_argument('--mode', default='train', type=str, 
                        help='train or test.')
    parser.add_argument('--amp_enable', action='store_true', default=False, 
                        help='use amp or not.')
    parser.add_argument('--wandb_enable', action='store_true', default=False, 
                        help='use wandb for result logging or not.')
    parser.add_argument('--resume', action='store_true', default=False, 
                        help='resume training or not.')
    parser.add_argument('--eval_ckpt', default='', type=str, 
                        help='checkpoint path for evaluation.')
    parser.add_argument('--class-names', default=['all'], type=str, nargs='+', 
                        help='class names for training')
    parser.add_argument('--lr', default=1e-4, type=float, 
                        help='learning rate')
    parser.add_argument('--batch-size', default=8, type=int, 
                        help='train batch size')
    parser.add_argument('--workers', default=c.workers, type=int,
                        help='num dataloader workers')
    parser.add_argument('--log-interval', default=c.log_interval, type=int,
                        help='log training progress every N batches (0 to disable)')
    parser.add_argument('--mstc-frame-stride', default=c.mstc_frame_stride, type=int,
                        help='frame stride for mstc (every Nth frame)')
    parser.add_argument('--meta-epochs', default=25, type=int,
                        help='number of meta epochs to train')
    parser.add_argument('--sub-epochs', default=4, type=int,
                        help='number of sub epochs to train')
    parser.add_argument('--extractor', default='wide_resnet50_2', type=str, 
                        help='feature extractor')
    parser.add_argument('--pool-type', default='avg', type=str, 
                        help='pool type for extracted feature maps')
    parser.add_argument('--parallel-blocks', default=[2, 5, 8], type=int, metavar='L', nargs='+',
                        help='number of flow blocks used in parallel flows.')
    parser.add_argument('--pro-eval', action='store_true', default=False, 
                        help='evaluate the pro score or not.')
    parser.add_argument('--pro-eval-interval', default=1, type=int, 
                        help='interval for pro evaluation.')
    parser.add_argument('--feature-cache', action='store_true', default=False,
                        help='use precomputed features instead of images (mstc only)')
    parser.add_argument('--feature-cache-dir', default=None, type=str,
                        help='root directory containing training/features and testing/features')
    parser.add_argument('--feature-subdir', default='features', type=str,
                        help='subdir name under split for feature cache')
    parser.add_argument('--feature-suffix', default='_res.npy', type=str,
                        help='feature file suffix (default: _res.npy)')
    parser.add_argument('--feature-fp32', dest='feature_fp32', action='store_true',
                        help='cast cached features to fp32 (default)')
    parser.add_argument('--feature-fp16', dest='feature_fp32', action='store_false',
                        help='keep cached features in fp16')
    parser.set_defaults(feature_fp32=True)
    parser.add_argument('--pixel-eval', dest='pixel_eval', action='store_true',
                        help='enable pixel-level eval (Loc.AUROC/PRO).')
    parser.add_argument('--no-pixel-eval', dest='pixel_eval', action='store_false',
                        help='disable pixel-level eval (Det.AUROC only).')
    parser.set_defaults(pixel_eval=None)
    parser.add_argument('--save-root', default=None, type=str,
                        help='override root directory for checkpoints (default: work_dir)')
    parser.add_argument(
        '--pruning-mode',
        default='dense',
        choices=[
            'dense',
            'sparse',
            'static',
            'dynamic',
            'reactivate_only',
            'kill_only',
            'kill_and_reactivate',
        ],
        help='pruning mode (legacy or DWA forward_type)',
    )
    parser.add_argument('--pruning-sparsity', type=float, default=0.0, help='global pruning sparsity [0,1]')
    parser.add_argument('--dwa-alpha', type=float, default=1.0, help='DWA alpha')
    parser.add_argument('--dwa-beta', type=float, default=1.0, help='DWA beta')
    parser.add_argument('--dwa-update-threshold', action='store_true', default=False, help='update DWA threshold once')
    parser.add_argument('--dwa-threshold-percentile', type=int, default=50, help='percentile for DWA threshold')

    args = parser.parse_args()

    for k, v in vars(args).items():
        setattr(c, k, v)

    if c.save_root:
        c.work_dir = c.save_root
    
    if c.dataset == 'mvtec':
        from datasets import MVTEC_CLASS_NAMES
        setattr(c, 'data_path', './data/MVTec')
        if c.class_names == ['all']:
            setattr(c, 'class_names', MVTEC_CLASS_NAMES)
    elif c.dataset == 'visa':
        from datasets import VISA_CLASS_NAMES
        setattr(c, 'data_path', './data/VisA_pytorch/1cls')
        if c.class_names == ['all']:
            setattr(c, 'class_names', VISA_CLASS_NAMES)
    elif c.dataset == 'mstc':
        from datasets import MSTC_CLASS_NAMES
        setattr(c, 'data_path', './data/shanghaitech')
        # mSTC isn’t multi-class; use one placeholder “shanghaitech”
        setattr(c, 'class_names', MSTC_CLASS_NAMES)
        
    if c.dataset == 'mvtec':
        c.input_size = (256, 256) if c.class_name == 'transistor' else (512, 512)
    elif c.dataset == 'mstc':
        c.input_size = (256, 384)

    pixel_eval_auto = c.pixel_eval is None
    if c.pixel_eval is None:
        c.pixel_eval = c.dataset != 'mstc'
    if c.feature_cache and pixel_eval_auto:
        c.pixel_eval = False
    if not c.pixel_eval:
        c.pro_eval = False

    return c

def main(c):
    c = parsing_args(c)
    init_seeds(seed=c.seed)
    c.version_name = 'msflow_{}_{}pool_pl{}'.format(c.extractor, c.pool_type, "".join([str(x) for x in c.parallel_blocks]))
    print(c.class_names)
    for class_name in c.class_names:
        c.class_name = class_name
        print('-+'*5, class_name, '+-'*5)
        c.ckpt_dir = os.path.join(c.work_dir, c.version_name, c.dataset, c.class_name)
        train(c)

if __name__ == '__main__':
    import default as c
    main(c)
