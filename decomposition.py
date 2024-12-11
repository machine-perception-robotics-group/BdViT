import os
import csv
import sys
import argparse
import torch
import torch.nn as nn
import time
import pathlib

from tqdm import tqdm
from timm.models import create_model
from collections import OrderedDict
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import utils
from decomposition_model import deit_base_patch16_224, deit_small_patch16_224, deit_tiny_patch16_224
from bdnn_module.utils import extract_weight_ext as ew
from bdnn_module.utils import calc_comput_complex as cFT


def args():
    parser = argparse.ArgumentParser('DeiT evaluation script', add_help=False)
    parser.add_argument('--img_path', default='~/data/imagenet/val_IF_WID', type=str, metavar='MODEL',
                        help='imagenet dataset path')
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--weights', default='./weights/deit_base/deit_base_patch16_224-b5f2ef4d.pth',
                        help='weights path')
    parser.add_argument('--device', default='cpu',
                        help='device to use for testing')
    parser.add_argument('--qb', default='8', type=int)
    return parser.parse_args()


def main(args):
    device = torch.device(args.device)

    print(f"Loading model: {args.model}")
    if 'deit_base' in args.model:
        model = deit_base_patch16_224(pretrained=True)
        weights_path = './weights/deit_base/deit_base_patch16_224-b5f2ef4d.pth'
    elif 'deit_small' in args.model:
        model = deit_small_patch16_224(pretrained=True)
        weights_path = './weights/deit_small/deit_small_patch16_224-cd65a155.pth'
    else:
        model = deit_tiny_patch16_224(pretrained=True)
        weights_path = './weights/deit_tiny/deit_tiny_patch16_224-a1311bcf.pth'
    
    save_dir = os.path.join(os.path.dirname(weights_path), 'binary7565_testttttttttttt')
    
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    range_basis = [args.qb]
    mode = 'exh'

    ew.save_param(model, range_basis, mode, save_dir, w_param = True)


if __name__ == '__main__':
    main(args())
