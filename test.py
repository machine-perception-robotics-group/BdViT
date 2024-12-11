import os
import csv
import sys
import argparse
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import time
import pathlib
import datetime

from tqdm import tqdm
from timm.models import create_model
from collections import OrderedDict
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import utils
from binary_model import deit_base_patch16_224, deit_small_patch16_224, deit_tiny_patch16_224

num_threads = 1


def args():
    parser = argparse.ArgumentParser('DeiT evaluation script', add_help=False)
    # Model parameters
    parser.add_argument('--img_path', default='ILSVRC2012_img_val_for_ImageFolder', type=str, metavar='MODEL',
                        help='imagenet dataset path')
    parser.add_argument('--model', default='deit_tiny_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--weights', default='./weights/deit_timy/deit_tiny_patch16_224-a1311bcf.pth',
                        help='weights path')
    parser.add_argument('--device', default='cpu',
                        help='device to use for testing')
    parser.add_argument('--qb', default='8', type=int)
    return parser.parse_args()


# calculating size of parameters [kB]
def calc_param_size(weight, dtype):
    weight_size = weight.size
    if weight_size >= 1:
        I = weight_size
        param_num = (I * dtype) / (8 * 1024 ^ 2)
    else:
        param_num = None
        print('weight shape must be 4 or 2 dimension.')

    return param_num


def text2list(file_path, name):
    match_list = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if name in row:
                match_list.append(row)

    return match_list


def calc_params(net, layer_lef, size_param):

    layer_ref = layer_lef
    list_size_bias_param = []  # Bias
    for name, param in net.state_dict().items():
        weight_dim = net.state_dict()[name].dim()
        weight_np = net.state_dict()[name].cpu().clone().detach().numpy()
        weight_np = weight_np.astype(np.float64)
        if (weight_np.size > 0) and (weight_dim == 1):
            list_size_bias_param.append(calc_param_size(weight_np, 32) / 1024.0)

    size_decomp_param = sum([x[3] for x in layer_ref]) # Decomposed parameters
    size_bias_param   = sum([x for x in list_size_bias_param])

    size_decomp_paramWbias = size_decomp_param + size_bias_param
    size_paramWbias = size_param + size_bias_param
    compress_ratio = 100 * (size_decomp_param / size_param)
    compress_ratioWbias = 100 * (size_paramWbias / size_decomp_paramWbias)

    print(f"TotalParams [MB]: {size_paramWbias: 4f}  DecomposedParams [MB]: {size_decomp_paramWbias: 4f}  Bias [MB]: {size_bias_param: 4f}")
    print(f"CompressRatio: {compress_ratio: 4f}  CompressRatioWbias: {compress_ratioWbias: 4f}")

    info_param = [['TotalParams [MB]', 'DecomposedParams [MB]', 'Bias [MB]', 'CompressRatio', 'CompressRatioWbias', 'TotalParamsWoutBias [MB]', 'DecomposedParamsWoutBias [MB]'],
                  [size_paramWbias, size_decomp_paramWbias, size_bias_param, compress_ratio, compress_ratioWbias, size_param, size_decomp_param]]

    return info_param


def load_param(net, Q_bits_list, layer_basis, param_dir, d_mode='exh'):
    basis_list = [os.path.join(param_dir, 'B' + str(k) + '/') for k in layer_basis]
    param_info_path = [os.path.join(param_dir, 'B' + str(k) + '/param_info_' + d_mode + '.csv') for k in layer_basis]
    total_param = float(text2list(param_info_path[0], 'ALL')[0][2])
    
    # net.patch_embed.proj.M_ui  = Parameter(torch.from_numpy(np.load(os.path.join(basis_list[48], 'patch_embed.proj.M_ui_exh.pth.npy')).astype(np.int64)), requires_grad=False)
    # net.patch_embed.proj.c     = Parameter(torch.from_numpy(np.load(os.path.join(basis_list[48], 'patch_embed.proj.c_exh.pth.npy'))))
    
    net.patch_embed.proj.M_ui_np  = np.load(os.path.join(basis_list[48], 'patch_embed.proj.M_ui_exh.pth.npy')).astype(np.uint64)
    net.patch_embed.proj.c_np     = np.load(os.path.join(basis_list[48], 'patch_embed.proj.c_exh.pth.npy')).astype(np.float64)

    net.blocks[0].attn.qkv.M_ui_np  = np.load(os.path.join(basis_list[24], 'blocks.0.attn.qkv.M_ui_exh.pth.npy'))
    net.blocks[0].attn.qkv.c_np     = np.load(os.path.join(basis_list[24], 'blocks.0.attn.qkv.c_exh.pth.npy'))
    net.blocks[0].attn.proj.M_ui_np = np.load(os.path.join(basis_list[25], 'blocks.0.attn.proj.M_ui_exh.pth.npy'))
    net.blocks[0].attn.proj.c_np    = np.load(os.path.join(basis_list[25], 'blocks.0.attn.proj.c_exh.pth.npy'))

    net.blocks[0].mlp.fc1.M_ui_np = np.load(os.path.join(basis_list[0], 'blocks.0.mlp.fc1.M_ui_exh.pth.npy'))
    net.blocks[0].mlp.fc1.c_np    = np.load(os.path.join(basis_list[0], 'blocks.0.mlp.fc1.c_exh.pth.npy'))
    net.blocks[0].mlp.fc2.M_ui_np = np.load(os.path.join(basis_list[1], 'blocks.0.mlp.fc2.M_ui_exh.pth.npy'))
    net.blocks[0].mlp.fc2.c_np    = np.load(os.path.join(basis_list[1], 'blocks.0.mlp.fc2.c_exh.pth.npy'))

    net.blocks[1].attn.qkv.M_ui_np  = np.load(os.path.join(basis_list[26], 'blocks.1.attn.qkv.M_ui_exh.pth.npy'))
    net.blocks[1].attn.qkv.c_np     = np.load(os.path.join(basis_list[26], 'blocks.1.attn.qkv.c_exh.pth.npy'))
    net.blocks[1].attn.proj.M_ui_np = np.load(os.path.join(basis_list[27], 'blocks.1.attn.proj.M_ui_exh.pth.npy'))
    net.blocks[1].attn.proj.c_np    = np.load(os.path.join(basis_list[27], 'blocks.1.attn.proj.c_exh.pth.npy'))

    net.blocks[1].mlp.fc1.M_ui_np = np.load(os.path.join(basis_list[2], 'blocks.1.mlp.fc1.M_ui_exh.pth.npy'))
    net.blocks[1].mlp.fc1.c_np    = np.load(os.path.join(basis_list[2], 'blocks.1.mlp.fc1.c_exh.pth.npy'))
    net.blocks[1].mlp.fc2.M_ui_np = np.load(os.path.join(basis_list[3], 'blocks.1.mlp.fc2.M_ui_exh.pth.npy'))
    net.blocks[1].mlp.fc2.c_np    = np.load(os.path.join(basis_list[3], 'blocks.1.mlp.fc2.c_exh.pth.npy'))

    net.blocks[2].attn.qkv.M_ui_np  = np.load(os.path.join(basis_list[28], 'blocks.2.attn.qkv.M_ui_exh.pth.npy'))
    net.blocks[2].attn.qkv.c_np     = np.load(os.path.join(basis_list[28], 'blocks.2.attn.qkv.c_exh.pth.npy'))
    net.blocks[2].attn.proj.M_ui_np = np.load(os.path.join(basis_list[29], 'blocks.2.attn.proj.M_ui_exh.pth.npy'))
    net.blocks[2].attn.proj.c_np    = np.load(os.path.join(basis_list[29], 'blocks.2.attn.proj.c_exh.pth.npy'))

    net.blocks[2].mlp.fc1.M_ui_np = np.load(os.path.join(basis_list[4], 'blocks.2.mlp.fc1.M_ui_exh.pth.npy'))
    net.blocks[2].mlp.fc1.c_np    = np.load(os.path.join(basis_list[4], 'blocks.2.mlp.fc1.c_exh.pth.npy'))
    net.blocks[2].mlp.fc2.M_ui_np = np.load(os.path.join(basis_list[5], 'blocks.2.mlp.fc2.M_ui_exh.pth.npy'))
    net.blocks[2].mlp.fc2.c_np    = np.load(os.path.join(basis_list[5], 'blocks.2.mlp.fc2.c_exh.pth.npy'))

    net.blocks[3].attn.qkv.M_ui_np  = np.load(os.path.join(basis_list[30], 'blocks.3.attn.qkv.M_ui_exh.pth.npy'))
    net.blocks[3].attn.qkv.c_np     = np.load(os.path.join(basis_list[30], 'blocks.3.attn.qkv.c_exh.pth.npy'))
    net.blocks[3].attn.proj.M_ui_np = np.load(os.path.join(basis_list[31], 'blocks.3.attn.proj.M_ui_exh.pth.npy'))
    net.blocks[3].attn.proj.c_np    = np.load(os.path.join(basis_list[31], 'blocks.3.attn.proj.c_exh.pth.npy'))

    net.blocks[3].mlp.fc1.M_ui_np = np.load(os.path.join(basis_list[6], 'blocks.3.mlp.fc1.M_ui_exh.pth.npy'))
    net.blocks[3].mlp.fc1.c_np    = np.load(os.path.join(basis_list[6], 'blocks.3.mlp.fc1.c_exh.pth.npy'))
    net.blocks[3].mlp.fc2.M_ui_np = np.load(os.path.join(basis_list[7], 'blocks.3.mlp.fc2.M_ui_exh.pth.npy'))
    net.blocks[3].mlp.fc2.c_np    = np.load(os.path.join(basis_list[7], 'blocks.3.mlp.fc2.c_exh.pth.npy'))

    net.blocks[4].attn.qkv.M_ui_np  = np.load(os.path.join(basis_list[32], 'blocks.4.attn.qkv.M_ui_exh.pth.npy'))
    net.blocks[4].attn.qkv.c_np     = np.load(os.path.join(basis_list[32], 'blocks.4.attn.qkv.c_exh.pth.npy'))
    net.blocks[4].attn.proj.M_ui_np = np.load(os.path.join(basis_list[33], 'blocks.4.attn.proj.M_ui_exh.pth.npy'))
    net.blocks[4].attn.proj.c_np    = np.load(os.path.join(basis_list[33], 'blocks.4.attn.proj.c_exh.pth.npy'))

    net.blocks[4].mlp.fc1.M_ui_np = np.load(os.path.join(basis_list[8], 'blocks.4.mlp.fc1.M_ui_exh.pth.npy'))
    net.blocks[4].mlp.fc1.c_np    = np.load(os.path.join(basis_list[8], 'blocks.4.mlp.fc1.c_exh.pth.npy'))
    net.blocks[4].mlp.fc2.M_ui_np = np.load(os.path.join(basis_list[9], 'blocks.4.mlp.fc2.M_ui_exh.pth.npy'))
    net.blocks[4].mlp.fc2.c_np    = np.load(os.path.join(basis_list[9], 'blocks.4.mlp.fc2.c_exh.pth.npy'))

    net.blocks[5].attn.qkv.M_ui_np  = np.load(os.path.join(basis_list[34], 'blocks.5.attn.qkv.M_ui_exh.pth.npy'))
    net.blocks[5].attn.qkv.c_np     = np.load(os.path.join(basis_list[34], 'blocks.5.attn.qkv.c_exh.pth.npy'))
    net.blocks[5].attn.proj.M_ui_np = np.load(os.path.join(basis_list[35], 'blocks.5.attn.proj.M_ui_exh.pth.npy'))
    net.blocks[5].attn.proj.c_np    = np.load(os.path.join(basis_list[35], 'blocks.5.attn.proj.c_exh.pth.npy'))

    net.blocks[5].mlp.fc1.M_ui_np = np.load(os.path.join(basis_list[10], 'blocks.5.mlp.fc1.M_ui_exh.pth.npy'))
    net.blocks[5].mlp.fc1.c_np    = np.load(os.path.join(basis_list[10], 'blocks.5.mlp.fc1.c_exh.pth.npy'))
    net.blocks[5].mlp.fc2.M_ui_np = np.load(os.path.join(basis_list[11], 'blocks.5.mlp.fc2.M_ui_exh.pth.npy'))
    net.blocks[5].mlp.fc2.c_np    = np.load(os.path.join(basis_list[11], 'blocks.5.mlp.fc2.c_exh.pth.npy'))

    net.blocks[6].attn.qkv.M_ui_np  = np.load(os.path.join(basis_list[36], 'blocks.6.attn.qkv.M_ui_exh.pth.npy'))
    net.blocks[6].attn.qkv.c_np     = np.load(os.path.join(basis_list[36], 'blocks.6.attn.qkv.c_exh.pth.npy'))
    net.blocks[6].attn.proj.M_ui_np = np.load(os.path.join(basis_list[37], 'blocks.6.attn.proj.M_ui_exh.pth.npy'))
    net.blocks[6].attn.proj.c_np    = np.load(os.path.join(basis_list[37], 'blocks.6.attn.proj.c_exh.pth.npy'))

    net.blocks[6].mlp.fc1.M_ui_np = np.load(os.path.join(basis_list[12], 'blocks.6.mlp.fc1.M_ui_exh.pth.npy'))
    net.blocks[6].mlp.fc1.c_np    = np.load(os.path.join(basis_list[12], 'blocks.6.mlp.fc1.c_exh.pth.npy'))
    net.blocks[6].mlp.fc2.M_ui_np = np.load(os.path.join(basis_list[13], 'blocks.6.mlp.fc2.M_ui_exh.pth.npy'))
    net.blocks[6].mlp.fc2.c_np    = np.load(os.path.join(basis_list[13], 'blocks.6.mlp.fc2.c_exh.pth.npy'))

    net.blocks[7].attn.qkv.M_ui_np  = np.load(os.path.join(basis_list[38], 'blocks.7.attn.qkv.M_ui_exh.pth.npy'))
    net.blocks[7].attn.qkv.c_np     = np.load(os.path.join(basis_list[38], 'blocks.7.attn.qkv.c_exh.pth.npy'))
    net.blocks[7].attn.proj.M_ui_np = np.load(os.path.join(basis_list[39], 'blocks.7.attn.proj.M_ui_exh.pth.npy'))
    net.blocks[7].attn.proj.c_np    = np.load(os.path.join(basis_list[39], 'blocks.7.attn.proj.c_exh.pth.npy'))

    net.blocks[7].mlp.fc1.M_ui_np = np.load(os.path.join(basis_list[14], 'blocks.7.mlp.fc1.M_ui_exh.pth.npy'))
    net.blocks[7].mlp.fc1.c_np    = np.load(os.path.join(basis_list[14], 'blocks.7.mlp.fc1.c_exh.pth.npy'))
    net.blocks[7].mlp.fc2.M_ui_np = np.load(os.path.join(basis_list[15], 'blocks.7.mlp.fc2.M_ui_exh.pth.npy'))
    net.blocks[7].mlp.fc2.c_np    = np.load(os.path.join(basis_list[15], 'blocks.7.mlp.fc2.c_exh.pth.npy'))

    net.blocks[8].attn.qkv.M_ui_np  = np.load(os.path.join(basis_list[40], 'blocks.8.attn.qkv.M_ui_exh.pth.npy'))
    net.blocks[8].attn.qkv.c_np     = np.load(os.path.join(basis_list[40], 'blocks.8.attn.qkv.c_exh.pth.npy'))
    net.blocks[8].attn.proj.M_ui_np = np.load(os.path.join(basis_list[41], 'blocks.8.attn.proj.M_ui_exh.pth.npy'))
    net.blocks[8].attn.proj.c_np    = np.load(os.path.join(basis_list[41], 'blocks.8.attn.proj.c_exh.pth.npy'))

    net.blocks[8].mlp.fc1.M_ui_np = np.load(os.path.join(basis_list[16], 'blocks.8.mlp.fc1.M_ui_exh.pth.npy'))
    net.blocks[8].mlp.fc1.c_np    = np.load(os.path.join(basis_list[16], 'blocks.8.mlp.fc1.c_exh.pth.npy'))
    net.blocks[8].mlp.fc2.M_ui_np = np.load(os.path.join(basis_list[17], 'blocks.8.mlp.fc2.M_ui_exh.pth.npy'))
    net.blocks[8].mlp.fc2.c_np    = np.load(os.path.join(basis_list[17], 'blocks.8.mlp.fc2.c_exh.pth.npy'))

    net.blocks[9].attn.qkv.M_ui_np  = np.load(os.path.join(basis_list[42], 'blocks.9.attn.qkv.M_ui_exh.pth.npy'))
    net.blocks[9].attn.qkv.c_np     = np.load(os.path.join(basis_list[42], 'blocks.9.attn.qkv.c_exh.pth.npy'))
    net.blocks[9].attn.proj.M_ui_np = np.load(os.path.join(basis_list[43], 'blocks.9.attn.proj.M_ui_exh.pth.npy'))
    net.blocks[9].attn.proj.c_np    = np.load(os.path.join(basis_list[43], 'blocks.9.attn.proj.c_exh.pth.npy'))

    net.blocks[9].mlp.fc1.M_ui_np = np.load(os.path.join(basis_list[18], 'blocks.9.mlp.fc1.M_ui_exh.pth.npy'))
    net.blocks[9].mlp.fc1.c_np    = np.load(os.path.join(basis_list[18], 'blocks.9.mlp.fc1.c_exh.pth.npy'))
    net.blocks[9].mlp.fc2.M_ui_np = np.load(os.path.join(basis_list[19], 'blocks.9.mlp.fc2.M_ui_exh.pth.npy'))
    net.blocks[9].mlp.fc2.c_np    = np.load(os.path.join(basis_list[19], 'blocks.9.mlp.fc2.c_exh.pth.npy'))

    net.blocks[10].attn.qkv.M_ui_np  = np.load(os.path.join(basis_list[44], 'blocks.10.attn.qkv.M_ui_exh.pth.npy'))
    net.blocks[10].attn.qkv.c_np     = np.load(os.path.join(basis_list[44], 'blocks.10.attn.qkv.c_exh.pth.npy'))
    net.blocks[10].attn.proj.M_ui_np = np.load(os.path.join(basis_list[45], 'blocks.10.attn.proj.M_ui_exh.pth.npy'))
    net.blocks[10].attn.proj.c_np    = np.load(os.path.join(basis_list[45], 'blocks.10.attn.proj.c_exh.pth.npy'))

    net.blocks[10].mlp.fc1.M_ui_np = np.load(os.path.join(basis_list[20], 'blocks.10.mlp.fc1.M_ui_exh.pth.npy'))
    net.blocks[10].mlp.fc1.c_np    = np.load(os.path.join(basis_list[20], 'blocks.10.mlp.fc1.c_exh.pth.npy'))
    net.blocks[10].mlp.fc2.M_ui_np = np.load(os.path.join(basis_list[21], 'blocks.10.mlp.fc2.M_ui_exh.pth.npy'))
    net.blocks[10].mlp.fc2.c_np    = np.load(os.path.join(basis_list[21], 'blocks.10.mlp.fc2.c_exh.pth.npy'))

    net.blocks[11].attn.qkv.M_ui_np  = np.load(os.path.join(basis_list[46], 'blocks.11.attn.qkv.M_ui_exh.pth.npy'))
    net.blocks[11].attn.qkv.c_np     = np.load(os.path.join(basis_list[46], 'blocks.11.attn.qkv.c_exh.pth.npy'))
    net.blocks[11].attn.proj.M_ui_np = np.load(os.path.join(basis_list[47], 'blocks.11.attn.proj.M_ui_exh.pth.npy'))
    net.blocks[11].attn.proj.c_np    = np.load(os.path.join(basis_list[47], 'blocks.11.attn.proj.c_exh.pth.npy'))

    net.blocks[11].mlp.fc1.M_ui_np = np.load(os.path.join(basis_list[22], 'blocks.11.mlp.fc1.M_ui_exh.pth.npy'))
    net.blocks[11].mlp.fc1.c_np    = np.load(os.path.join(basis_list[22], 'blocks.11.mlp.fc1.c_exh.pth.npy'))
    net.blocks[11].mlp.fc2.M_ui_np = np.load(os.path.join(basis_list[23], 'blocks.11.mlp.fc2.M_ui_exh.pth.npy'))
    net.blocks[11].mlp.fc2.c_np    = np.load(os.path.join(basis_list[23], 'blocks.11.mlp.fc2.c_exh.pth.npy'))

    net.head.M_ui_np = np.load(os.path.join(basis_list[49], 'head.M_ui_exh.pth.npy'))
    net.head.c_np    = np.load(os.path.join(basis_list[49], 'head.c_exh.pth.npy'))

    print('+-----Load parameters-----+')
    layer_info = []
    cnt = 0
    for name, param in net.state_dict().items():
        if '.' not in name:
            continue
        layer_split = name.split('.')
        name_len = len(layer_split)
        _, param_name_re = name.rsplit('.', 1)
        
        if name_len == 2:
            layer_name, param_name = layer_split
            layer_rename = layer_name
            layer_param_name = name
        elif name_len == 3:
            conv_name, conv_idx, param_name = layer_split
            if type(conv_idx) is str:
                layer_rename = '{}.{}'.format(conv_name, conv_idx)
            else:
                layer_rename = '{}[{}]'.format(conv_name, conv_idx)
            layer_param_name = layer_rename + '.' + param_name
        elif name_len == 4:
            conv_name, conv_idx, func_name, param_name = layer_split
            layer_rename = '{}[{}].{}'.format(conv_name, conv_idx, func_name)
            layer_param_name = layer_rename + '.' + param_name
        elif name_len == 5:
            conv_name, conv_idx, func_name, func_idx, param_name = layer_split
            layer_rename = '{}[{}].{}.{}'.format(conv_name, conv_idx, func_name, func_idx)
            layer_param_name = layer_rename + '.' + param_name
        else:
            conv_name, conv_idx, func_name, func_idx, option_name, param_name = layer_split
            layer_rename = '{}[{}].{}.{}.{}'.format(conv_name, conv_idx, func_name, func_idx, option_name)
            layer_param_name = layer_rename + '.' + param_name

        if 'weight' == param_name_re:
            weight_dim = net.state_dict()[name].dim()
            if (weight_dim == 2) or (weight_dim == 4):
                d_param = float(text2list(param_info_path[cnt], name)[0][3])
                layer_info.append([layer_rename, Q_bits_list[cnt], layer_basis[cnt], d_param])
                print('{}\tquantize_bits={}  basis={}  d_param={:.4f}[MB]'.format(
                    layer_rename, Q_bits_list[cnt], layer_basis[cnt], d_param))
                cnt += 1

        if ('M_ui' == param_name_re) or ('c' == param_name_re):
            exec("net.{}.basis = layer_basis[cnt - 1]".format(layer_rename))
            exec("net.{}.quantize_bits = Q_bits_list[cnt - 1]".format(layer_rename))
            exec("net.{}.num_threads = {}".format(layer_rename, num_threads))

        if 'weight_approx' == param_name_re:
            new_param = np.load(os.path.join(basis_list[cnt - 1], name + '.npy')).astype(np.float32)
            new_param = Parameter(torch.from_numpy(new_param), requires_grad=False)
            exec("net.{} = new_param".format(layer_param_name))

    print(f'+---{cnt} parameters loaded---+')
    info_param = calc_params(net, layer_info, total_param)
    return info_param


def to_csv(csv_list, path_file):
    with open(path_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluation(val_loder, model, device, args):
    print("Starting evaluation")
    model.eval()

    Acc1, Acc5 = 0, 0
    ave_time: float = 0
    size_data = len(val_loder)
    with torch.no_grad():
        with tqdm(enumerate(val_loder), total=size_data, desc='iteration') as pbar:
            for i, (image, target) in pbar:
                image,  target = image.to(device), target.to(device)

                start_eval = time.time()
                output = model(image)
                end_eval = time.time()
                output = nn.functional.softmax(output, dim=1)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                Acc1 += acc1.item()
                Acc5 += acc5.item()
                net_time = (end_eval - start_eval)
                ave_time += net_time
                pbar.set_postfix(OrderedDict(net_time=f'{net_time:.4f}', acc=f'{acc1.item():3.1f}'))

    ave_time /= len(val_loder)
    top1, top5 = Acc1/size_data, Acc5/size_data
    num_params = sum(p.numel() for p in model.parameters())
    save_list = [['Top1Acc', 'Top5Acc', 'Top1Err', 'Top5Err', 'AveTime', 'NumOfParams'],
                 [top1, top5, 100.0-top1, 100.0-top5, ave_time, num_params]]

    print(f"Top1Acc: {top1:.4f}  Top5Acc: {top5:.4f}")
    print(f"Average forward time: {ave_time:.4f}  Parameter numbers: {num_params}")

    return save_list


def evaluation_1000(val_loder, model, device, args):
    print("Starting evaluation")
    model.eval()

    Acc1, Acc5 = 0, 0
    ave_time: float = 0
    size_data = 1000
    with torch.no_grad():
        with tqdm(enumerate(val_loder), total=size_data, desc='iteration') as pbar:
            for i, (image, target) in pbar:
                image,  target = image.to(device), target.to(device)

                start_eval = time.time()
                output = model(image)
                end_eval = time.time()
                output = nn.functional.softmax(output, dim=1)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                Acc1 += acc1.item()
                Acc5 += acc5.item()
                net_time = (end_eval - start_eval)
                ave_time += net_time
                pbar.set_postfix(OrderedDict(net_time=f'{net_time:.4f}', acc=f'{acc1.item():3.1f}'))
                if i+1 >= 1000:
                    break

    ave_time /= size_data
    top1, top5 = Acc1/size_data, Acc5/size_data
    num_params = sum(p.numel() for p in model.parameters())
    save_list = [['Top1Acc', 'Top5Acc', 'Top1Err', 'Top5Err', 'AveTime', 'NumOfParams'],
                 [top1, top5, 100.0-top1, 100.0-top5, ave_time, num_params]]

    print(f"Top1Acc: {top1:.4f}  Top5Acc: {top5:.4f}")
    print(f"Average forward time: {ave_time:.4f}  Parameter numbers: {num_params}")

    return save_list


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
    model.to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device)['model'], strict=False)



    range_q_bits_all = [args.qb for i in range(50)]
    range_basis_all  = [args.qb for i in range(50)]
    
    
    dir_param = os.path.join(os.path.dirname(weights_path), 'binary7565')
    info_param = load_param(model, range_q_bits_all, range_basis_all, dir_param, d_mode='exh')

    transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    valset = ImageFolder(root=args.img_path, transform=transform)
    val_loader  = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)

    #info_result =  evaluation(val_loader, model, device, args)
    info_result = evaluation_1000(val_loader, model, device, args)
    info_param.extend(info_result)

    path_file, _ = os.path.splitext(weights_path)
    os.makedirs(path_file, exist_ok=True)
    
    current_datetime = datetime.datetime.now()
    datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")
    filename = f"R_B6Q8_7565_50000_float_{datetime_str}.csv"
    
    to_csv(info_param, os.path.join(path_file, filename))

if __name__ == '__main__':
    main(args())
