from train import Trainer
import argparse
from config.config import cfg, cfg_from_file
from easydict import EasyDict as edict
import pprint
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='Train a UNet...')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/config.yaml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)
    
    t = Trainer(cfg.DATA_DIR, '', cfg.MODEL_DIR, img_save_path='./train_image', split='train', bz = cfg.BATCH_SIZE, numworks = cfg.WORKERS, max_epoch=cfg.MAX_EPOCH, lr=cfg.LR)
    t.train(cfg.MAX_EPOCH)