import torch
import torch.nn as nn
from torchvision.transforms import *
import torch.nn.functional as F

import numpy as np

from torchvision import models

import cv2
from PIL import Image

from module.transformer import TransformerEncoderLayer
from module.position_encoding import PositionEmbeddingSine
from module.spade import SPADE

def visual(image):
    im =(image).transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()

def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

class ConvWithActivation_encode(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(ConvWithActivation_encode, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x), x
        else:
            return x

class ConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(ConvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class DeConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization deconv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, output_padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(DeConvWithActivation, self).__init__()
        self.conv2d = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,output_padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class LateralConnect(nn.Module):
    def __init__(self, inputChannels, outputChannels):
        super(LateralConnect, self).__init__()
        mid_channel = int(inputChannels/2)
        self.con1 = ConvWithActivation_encode(inputChannels, mid_channel, 3, 1, 1)
        self.con2 = ConvWithActivation_encode(mid_channel, mid_channel, 3, 1, 1)
        self.con3 = ConvWithActivation_encode(mid_channel, outputChannels, 3, 1, 1)
    
    def forward(self,x):
        x,_ = self.con1(x)
        x,_ = self.con2(x)
        x,_ = self.con3(x)
        return x

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
      #  vgg16.load_state_dict(torch.load('./vgg16-397923af.pth'))
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, in_channels, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning

class ConvTD_SPADE_refine(nn.Module):
    def __init__(self, input_channels, residual_blocks):
        super(ConvTD_SPADE_refine, self).__init__()

        self.G0 = 256
        self.G = 32
        self.D = 16
        self.C = 6

        ### encoder-decoder ###
        self.ec1 = ConvWithActivation_encode(7,64,4,2,1)
        self.res1 = ResnetBlock(64, dilation=1, use_spectral_norm=False)
        self.encoder_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.ec2 = ConvWithActivation_encode(128,128,4,2,1)
        self.res2 = ResnetBlock(128, dilation=1, use_spectral_norm=False)
        self.encoder_conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)   

        blocks = []
        for _ in range(residual_blocks):
            block = ConvTE_spade(256, layout_dim=256, dilation=2, use_spectral_norm=False, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.dc1_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dc1 = DeConvWithActivation(256,64,4,2,1)   
        self.dc2_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dc2 = DeConvWithActivation(128,64,4,2,1)  
        self.dc3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
            )

        self.tanh = nn.Tanh()

        self.lc6 = LateralConnect(128,128)
        self.lc7 = LateralConnect(64,64)

        self.prejection_head1 = nn.Conv2d(128,3,kernel_size=1)
        self.prejection_head2 = nn.Conv2d(64,3,kernel_size=1)

        # Encoder semantic branch   from SPL
        self.encoder_prePad_sm = nn.ReflectionPad2d(3)
        self.encoder_conv1_sm = nn.Conv2d(in_channels=input_channels+1, out_channels=64, kernel_size=7, padding=0)
        self.encoder_relu1_sm = nn.LeakyReLU(0.1)
        self.encoder_conv2_sm = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.encoder_relu2_sm = nn.LeakyReLU(0.1)
        self.encoder_conv3_sm = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.encoder_relu3_sm = nn.LeakyReLU(0.1)
        self.encoder_conv4_sm = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.encoder_relu4_sm = nn.LeakyReLU(0.1)

        self.encoder_sm_out = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        # branch for Asl feature recon
        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G0, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G0 * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2),
        )
        self.feature_recon = nn.Sequential(
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
            ResnetBlock(256, dilation=1, use_spectral_norm=False),
        )

        self.feature_mapping = nn.Conv2d(in_channels=256, out_channels=152, kernel_size=1, stride=1, padding=0)

    def encoder_sm(self, x):
        x = self.encoder_prePad_sm(x)

        x = self.encoder_conv1_sm(x)
        x = self.encoder_relu2_sm(x)

        x = self.encoder_conv2_sm(x)
        x = self.encoder_relu2_sm(x)

        x = self.encoder_conv3_sm(x)
        x = self.encoder_relu3_sm(x)

        x = self.encoder_conv4_sm(x)
        x = self.encoder_relu4_sm(x)

        return x
        

    def forward(self, x, mask, soft_mask, coarse_out):
        x_input = x 

        ### encoder ###
        # import pdb;pdb.set_trace()
        ef1, skipConnect1 = self.ec1(torch.cat((x_input, coarse_out, soft_mask),1))
        # ef1, skipConnect1 = self.ec1(x_input)
        ef1 = self.res1(ef1)
        ef1 = self.encoder_conv1(ef1)

        ef2, skipConnect2 = self.ec2(ef1)
        ef2 = self.res2(ef2)
        ef2 = self.encoder_conv2(ef2)
        ### encoder ###

        ### semantic encoder ###   from SPL
        x_sm = self.encoder_sm(torch.cat((x_input, mask), 1))
        x_sm_skip = self.encoder_sm_out(x_sm)
        local_features = []
        for i in range(self.D):
            x_sm_skip = self.rdbs[i](x_sm_skip)
            local_features.append(x_sm_skip)

        x_sm = self.gff(torch.cat(local_features, 1)) + x_sm
        layout = self.feature_recon(x_sm)
        feature_recon = self.feature_mapping(layout)        
        ### semantic encoder ###

        ### transformer ###
        for i in range(len(self.middle)):
            sub_block = self.middle[i]
            ef2 = sub_block(ef2, layout)
        ### transformer ###

        ### decoder ###  
        deFeatures2 = self.dc1_conv(ef2)
        concatMap2 = torch.cat((deFeatures2, self.lc6(skipConnect2)), 1)
        deFeatures3 = self.dc1(concatMap2)

        deFeatures3 = self.dc2_conv(deFeatures3)
        concatMap3 = torch.cat((deFeatures3, self.lc7(skipConnect1)),1)  
        deFeatures4 = self.dc2(concatMap3)  
      
        deFeatures5 = self.dc3(deFeatures4)

        output_128 = self.prejection_head1(deFeatures2)
        output_256 = self.prejection_head2(deFeatures3)

        out1 = (self.tanh(output_128) + 1) / 2
        out2 = (self.tanh(output_256) + 1) / 2
        output = (self.tanh(deFeatures5) + 1) / 2
        ### decoder ###   
        return out1, out2, output, feature_recon

class StructureGen(nn.Module):
    def __init__(self, n_in_channel=3):
        super(StructureGen, self).__init__()
        #downsample
        self.conv1 = ConvWithActivation(4,32,kernel_size=4,stride=2,padding=1)
        self.conva = ConvWithActivation(32,32,kernel_size=3, stride=1, padding=1)
        self.convb = ConvWithActivation(32,64, kernel_size=4, stride=2, padding=1)
        self.res1 = Downsample_connect(64,64)
        self.res2 = Downsample_connect(64,64)
        self.res3 = Downsample_connect(64,128,same_shape=False)
        self.res4 = Downsample_connect(128,128)
        self.res5 = Downsample_connect(128,256,same_shape=False)
       # self.nn = ConvWithActivation(256, 512, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
        self.res6 = Downsample_connect(256,256)
        self.res7 = Downsample_connect(256,512,same_shape=False)
        self.res8 = Downsample_connect(512,512)
        self.conv2 = ConvWithActivation(512,512,kernel_size=1)

        #upsample
        self.deconv1 = DeConvWithActivation(512,256,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.deconv2 = DeConvWithActivation(256*2,128,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.deconv3 = DeConvWithActivation(128*2,64,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.deconv4 = DeConvWithActivation(64*2,32,kernel_size=3,padding=1,stride=2,output_padding=1)
        self.deconv5 = DeConvWithActivation(64,3,kernel_size=3,padding=1,stride=2,output_padding=1)

        #lateral connection 
        self.lateral_connection1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(512, 512, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(512, 256, kernel_size=1, padding=0,stride=1),)
        self.lateral_connection2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(256, 128, kernel_size=1, padding=0,stride=1),)
        self.lateral_connection3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(128, 64, kernel_size=1, padding=0,stride=1),)
        self.lateral_connection4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, padding=0,stride=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1,stride=1),
            nn.Conv2d(64, 32, kernel_size=1, padding=0,stride=1),)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conva(x)
        con_x1 = x
       # import pdb;pdb.set_trace()
        x = self.convb(x)
        x = self.res1(x)
        con_x2 = x
        x = self.res2(x)
        x = self.res3(x)
        con_x3 = x
        x = self.res4(x)
        x = self.res5(x)
        con_x4 = x
        x = self.res6(x)

       # import pdb;pdb.set_trace()
        x = self.res7(x)
        x = self.res8(x)
        x = self.conv2(x)
        #upsample
        x = self.deconv1(x)

        x = torch.cat([self.lateral_connection1(con_x4), x], dim=1)
        x = self.deconv2(x)
        x = torch.cat([self.lateral_connection2(con_x3), x], dim=1)
        x = self.deconv3(x)
        x = torch.cat([self.lateral_connection3(con_x2), x], dim=1)
        x = self.deconv4(x)
        x = torch.cat([self.lateral_connection4(con_x1), x], dim=1)
        x = self.deconv5(x) 
        return x            

# original D
class D_Net(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True):
        super(D_Net, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )


    def forward(self, x):

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class ConvTE_spade(nn.Module):
    def __init__(self, dim, layout_dim, dilation=1, use_spectral_norm=True, d_model=256, nhead=8, dim_feedforward=512, dropout=0.1):
        super(ConvTE_spade, self).__init__()
        self.TE = TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=0.1)
        self.spade = ResnetBlock_Spade(dim, layout_dim, dilation=1, use_spectral_norm=True)
        self.position_encode = PositionEmbeddingSine(128, normalize=True)

        self.ec1 = ConvWithActivation_encode(256,256,4,2,1)
        self.res1 = ResnetBlock(256, dilation=1, use_spectral_norm=False)
        self.ec2 = ConvWithActivation_encode(256,256,4,2,1)
        self.res2 = ResnetBlock(256, dilation=1, use_spectral_norm=False)  

        self.dc1 = DeConvWithActivation(512,256,4,2,1)   
        self.dc2 = DeConvWithActivation(512,256,4,2,1) 

        self.lc1 = LateralConnect(256,256)
        self.lc2 = LateralConnect(256,256)

    def forward(self, x, layout):
        spade_out = self.spade(x, layout)

        ## encoder
        x, skip1 = self.ec1(x)
        x = self.res1(x)

        x, skip2 = self.ec2(x)
        x = self.res2(x)

        ### transformer
        src = x
        bs, c, h, w = src.shape
        pos_embed, pad_mask = self.position_encode(src)
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2,0,1)
        te_out = self.TE(src, pos=pos_embed)
        te_out = te_out.permute(1, 2, 0).view(bs, c, h, w)

        ### decoder
        te_out = self.dc1(torch.cat((te_out, self.lc1(skip2)),1))
        te_out = self.dc2(torch.cat((te_out, self.lc1(skip1)),1))

        ### final_out
        out = spade_out + te_out
        return out

class Downsample_connect(nn.Module):
    def __init__(self, in_channels, out_channels, same_shape=True, **kwargs):
        super(Downsample_connect,self).__init__()
        self.same_shape = same_shape
        strides = 1 if same_shape else 2
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,stride=strides)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv2 = torch.nn.utils.spectral_norm(self.conv2)
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
            # self.conv3 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                 stride=strides)
            # self.conv3 = torch.nn.utils.spectral_norm(self.conv3)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        if not self.same_shape:
            x = self.conv3(x)
        out = self.batch_norm2d(out + x)
        # out = out + x
        return F.relu(out)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out

class ResnetBlock_Spade(nn.Module):
    def __init__(self, dim, layout_dim, dilation, use_spectral_norm=True):
        super(ResnetBlock_Spade, self).__init__()
        self.conv_block = nn.Sequential(
            SPADE (dim, layout_dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),

            SPADE(256, layout_dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(dim, track_running_stats=False),
            # RN_L(feature_channels=dim, threshold=threshold),
        )

    def forward(self, x, layout):
        # out = x + self.conv_block(x)
        out = x
        for i in range(len(self.conv_block)):
            sub_block = self.conv_block[i]
            if i == 0 or i == 4:
                out = sub_block(out, layout)
            else:
                out = sub_block(out)
    
        out_final = out + x
        # skimage.io.imsave('block.png', out[0].detach().permute(1,2,0).cpu().numpy()[:,:,0])

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out_final


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


if __name__ == '__main__':
    print("No Abnormal!")
