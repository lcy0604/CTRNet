import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch import autograd
import torchvision.models as models
from networks_transformer import D_Net, VGG16FeatureExtractor, ConvTD_SPADE_refine, StructureGen
from src.models import create_model

import cv2
from PIL import Image

def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t

def get_results(output):
    bboxes = output['bboxes']
    gt_kernels = []
    gt_kernel = np.zeros((512,512), dtype='uint8')
    if len(bboxes) > 0:
        for i in range(len(bboxes)):
            bboxes[i] = np.reshape(bboxes[i], (bboxes[i].shape[0] // 2, 2)).astype('int32')
        for i in range(len(bboxes)):
            cv2.drawContours(gt_kernel, [bboxes[i]], -1, 1, -1)
    gt_kernels.append(gt_kernel)
    gt_kernels = np.array(gt_kernels)
    det_mask = torch.from_numpy(gt_kernels).float().unsqueeze(0)
    return det_mask

def visual(image):
    im =(image).transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()

class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label).cuda())
        self.register_buffer('fake_label', torch.tensor(target_fake_label).cuda())

        # self.register_buffer('real_label', torch.tensor(target_real_label))   ### original code
        # self.register_buffer('fake_label', torch.tensor(target_fake_label))   ### original code

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class G_Net(nn.Module):
    def __init__(self, structure_path):
        super(G_Net, self).__init__()

        self.coarse_gen = StructureGen()
        if structure_path is not None:
            state_gen = torch.load(structure_path)
            self.coarse_gen.load_state_dict(state_gen)

        self.texture_generator = ConvTD_SPADE_refine(input_channels=3, residual_blocks=8)

    def forward(self, x, mask, soft_mask, structure_im):
        coarse_output = self.coarse_gen(torch.cat((structure_im,soft_mask),1))
        # import pdb;pdb.set_trace()
        out1, out2, prediction, img_f_pred = self.texture_generator(x, mask, soft_mask, coarse_output)
        return coarse_output, out1, out2, prediction, img_f_pred

class CTRNet(nn.Module):
    def __init__(self, g_lr, d_lr, l1_weight, gan_weight, TRresNet_path=None, Structure_path=None):
        super(CTRNet, self).__init__()

        self.generator = G_Net(structure_path=Structure_path)
        self.discriminator = D_Net(in_channels=3, use_sigmoid=True)

        self.extractor = VGG16FeatureExtractor()

        if TRresNet_path is not None:
            # import pretrained tresnet_xL
            state_xL = torch.load(TRresNet_path, map_location='cpu')
            pretrained_model = state_xL['model']
            self.tresnet_xL_hold = create_model('tresnet_l')
            model_dict = self.tresnet_xL_hold.state_dict()
            new_dict = {k: v for k, v in pretrained_model.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            self.tresnet_xL_hold.load_state_dict(model_dict)
            for k, v in self.tresnet_xL_hold.named_parameters():
                v.requires_grad=False
            # import pdb;pdb.set_trace()

        self.l1_loss = nn.L1Loss()
        self.l1_loss_feature = nn.L1Loss(reduction='none')
        self.adversarial_loss = AdversarialLoss('nsgan')

        self.g_lr, self.d_lr = g_lr, d_lr
        self.l1_weight, self.gan_weight = l1_weight, gan_weight

    def make_optimizer(self):
        self.gen_optimizer = optim.Adam(
            [{'params': self.generator.parameters(), 'lr': float(self.g_lr)}],
            lr=float(self.g_lr),
            betas=(0., 0.9)
        )

        self.dis_optimizer = optim.Adam(
            params=self.discriminator.parameters(),
            lr=float(self.d_lr),
            betas=(0., 0.9)
        )
