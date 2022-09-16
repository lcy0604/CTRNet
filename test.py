from __future__ import print_function
import argparse
from math import log10
import numpy as np
import math

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset import build_dataloader
import pdb
import socket
import time

from skimage import io

from models_CTRNet import CTRNet
from PIL import Image
import cv2

# Testing settings
parser = argparse.ArgumentParser(description='CTRNet_test')
parser.add_argument('--bs', type=int, default=64, help='training batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--cpu', default=False, action='store_true', help='Use CPU to test')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=67454, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--img_flist', type=str, default='/data/dataset/places2/flist/val.flist')
parser.add_argument('--model', default='./checkpoints', help='sr pretrained base model')
parser.add_argument('--save', default=False, action='store_true', help='If save test images')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--input_size', type=int, default=512, help='input image size')
parser.add_argument('--l1_weight', type=float, default=1.0)
parser.add_argument('--gan_weight', type=float, default=0.1)

opt = parser.parse_args()

def visual(image):
    im =(image).transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()

def eval(device):
    model.eval()
    for batch in testing_data_loader:
        img_512_batch, gt_batch, structure_im, structure_lbl, gt_text, soft_mask, index, name = batch
        t_io2 = time.time()
        if cuda:
            gt_batch = gt_batch.cuda(device)
            img_512_batch = img_512_batch.cuda(device)

            gt_text = gt_text.cuda(non_blocking=True)
            gt_text = gt_text.unsqueeze(1).cuda(non_blocking=True)

            soft_mask = soft_mask.unsqueeze(1).cuda(non_blocking=True)
            # import pdb;pdb.set_trace()
            structure_im = structure_im.cuda(non_blocking=True)
            mask_batch = soft_mask

        with torch.no_grad():
            mask_512 =  mask_batch #F.interpolate(mask_batch, 512)
            img_512_masked = img_512_batch * (1.0 - mask_batch) + mask_batch

            structure_output, out1, out2, prediction, img_f_pred = model.generator(img_512_batch, gt_text.float(), soft_mask, structure_im)

            withMask_prediction = prediction * mask_batch + img_512_batch * (1 - mask_batch)
            output = withMask_prediction

        if opt.save:
            str_path = opt.save_path + '/str_re/'
            withMask_path = opt.save_path + '/mask_re/'
            if not os.path.exists(str_path):
                os.mkdir(str_path)
                os.mkdir(withMask_path)     
            prediction = (prediction.detach().permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
            withMask_prediction = (withMask_prediction.detach().permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
            print(name[0].split('/')[-1].split('.')[0])     
            save_img(str_path, 'pred_'+name[0].split('/')[-1].split('.')[0], prediction[0])
            save_img(withMask_path, 'pred_'+name[0].split('/')[-1].split('.')[0], withMask_prediction[0])

def save_img(path, name, img):
    # img (H,W,C) or (H,W) np.uint8
    io.imsave(path+'/'+name+'.png', img)


if __name__ == '__main__':
    if opt.cpu:
        print("===== Use CPU to Test! =====")
    else:
        print("===== Use GPU to Test! =====")
    
    ## Set the GPU mode
    gpus_list=[0] #range(opt.gpus)
    cuda = not opt.cpu
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    model = CTRNet(g_lr=opt.lr, d_lr=(opt.lr), l1_weight=opt.l1_weight, gan_weight=opt.gan_weight)
    model = model.cuda()
    
    pretrained_model = torch.load(opt.model)

    if cuda:
        device = torch.device('cuda:0')
        model = model.cuda(device)
        # import pdb;pdb.set_trace()
        if len(gpus_list) > 1:
            model.generator = torch.nn.DataParallel(model.generator, device_ids=gpus_list)
            model.discriminator = torch.nn.DataParallel(model.discriminator, device_ids=gpus_list)
            model.load_state_dict(pretrained_model)
        else:
            state_dict = model.state_dict()
            new_dict_no_module = {}
            for k, v in pretrained_model.items():
                k = k.replace('module.', '')
                new_dict_no_module[k] = v

            new_dict = {k: v for k, v in new_dict_no_module.items() if k in state_dict.keys()}
            state_dict.update(new_dict)
            model.load_state_dict(state_dict)
    
    print(opt.model)
    print('Pre-trained G model is loaded.')

    # Datasets
    print('===> Loading datasets')
    testing_data_loader = build_dataloader(
        flist=opt.img_flist,
        training=False,
        input_size=opt.input_size,
        batch_size=opt.bs,
        num_workers=opt.threads,
        shuffle=False
    )
    print('===> Loaded datasets')

    ## Eval Start!!!!
    eval(device)
