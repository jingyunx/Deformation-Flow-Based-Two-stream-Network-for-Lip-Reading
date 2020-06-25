import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math
import torch.nn.functional as F

from lr_models.ResNet18 import ResNet_18
from lr_models.ResNet18_real import ResNet_18 as ResNet_18_real
from lr_models.ResNet18_real_bn import ResNet_18 as ResNet_18_real_bn
from lr_models.GRU import gen_GRU, gen_GRU_fab
from .models_multiview import FrontaliseModelMasks_wider

def initialize_weights(m):
    for m in m.modules():
        if isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class Encoder_W_real(nn.Module):
    def __init__(self, options):
        super(Encoder_W_real, self).__init__()
        self.frontend3D = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                )
        self.resnet18 = ResNet_18_real()
        self.dropout = nn.Dropout(p=0.5)
        self.apply(initialize_weights)

    def forward(self, input):
        x = input.transpose(1,2).contiguous()
        x = self.frontend3D(x)
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(-1, 64, x.size(3), x.size(4))
        x= self.resnet18(x)
        x = self.dropout(x)
        x = x.view(-1, 29, 512)
        return x

class Encoder_W_df(nn.Module):
    def __init__(self, options):
        super(Encoder_W_df, self).__init__()
        self.frontend2D = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=(7, 7), stride=( 2, 2), padding=(3, 3), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                )
        self.resnet18 = ResNet_18_real_bn()
        self.dropout = nn.Dropout(p=0.5)
        self.apply(initialize_weights)

    def forward(self, input):
        batch_size = input.size(0)
        x = input.view(-1, 2, input.size(3), input.size(4))
        x = self.frontend2D(x)
        x= self.resnet18(x)
        x = self.dropout(x)
        x = x.view(batch_size, 29, 512)
        return x

class Encoder_fab(nn.Module):
    def __init__(self, options):
        super(Encoder_fab, self).__init__()
        self.frontend2D = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(7, 7), stride=( 2, 2), padding=(3, 3), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                )
        self.resnet18 = ResNet_18()
        self.dropout = nn.Dropout(p=0.5)
        self.apply(initialize_weights)

    def forward(self, input):
        batch_size = input.size(0)
        num_frame = input.size(1)
        x = input.view(-1, 1, input.size(3), input.size(4))
        x = self.frontend2D(x)       
        x = self.resnet18(x)
        x = x.view(batch_size, num_frame, 256)
        return x

class FAB(nn.Module):
    def __init__(self, input_nc, num_decoders=5, inner_nc=256, num_additional_ids=32, smaller=False, num_masks=0):
        super(FAB, self).__init__()
        print(num_additional_ids, inner_nc)
        self.encoder = self.generate_encoder_layers(input_nc, output_size=inner_nc, num_filters=num_additional_ids)
        
        self.decoder = self.generate_decoder_layers(inner_nc*2, num_filters=num_additional_ids)

    def parallel_init(self):
        self.encoder =  nn.DataParallel(self.encoder).cuda()
        self.decoder =  nn.DataParallel(self.decoder).cuda()

    def generate_encoder_layers(self, input_nc, output_size=256, num_filters=32):
        conv1 = nn.Conv2d(input_nc, num_filters, 4, 2, 1)
        conv2 = nn.Conv2d(num_filters, num_filters * 2, 4, 2, 1)
        conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, 4, 2, 1)
        conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, 4, 2, 1)
        conv5 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        conv6 = nn.Conv2d(num_filters * 8, num_filters * 8, 4, 2, 1)
        conv7 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)  #可以试试512
        # conv8 = nn.Conv2d(num_filters * 8, output_size, 4, 2, 1)

        batch_norm = nn.BatchNorm2d(num_filters)
        batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
        batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
        batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
        # batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)

        leaky_relu = nn.LeakyReLU(0.2, True)
        return nn.Sequential(conv1, leaky_relu, conv2, batch_norm2_0, \
                              leaky_relu, conv3, batch_norm4_0, leaky_relu, \
                              conv4, batch_norm8_0, leaky_relu, conv5, 
                              batch_norm8_1, leaky_relu, conv6, batch_norm8_2, 
                              leaky_relu, conv7)

    def generate_decoder_layers(self, num_input_channels, num_output_channels=2, num_filters=32):
        up = nn.Upsample(scale_factor=2, mode='bilinear')

        dconv1 = nn.Conv2d(num_input_channels, num_filters*8, 3, 1, 1)
        dconv2 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
        dconv3 = nn.Conv2d(num_filters*8, num_filters*8, 3, 1, 1)
        dconv4 = nn.Conv2d(num_filters * 8 , num_filters * 4, 3, 1, 1)
        dconv5 = nn.Conv2d(num_filters * 4 , num_filters * 2, 3, 1, 1)
        dconv6 = nn.Conv2d(num_filters * 2 , num_filters , 3, 1, 1)
        dconv7 = nn.Conv2d(num_filters, num_output_channels, 3, 1, 1)
        # dconv8 = nn.Conv2d(num_filters , num_output_channels, 3, 1, 1)

        batch_norm1_0 = nn.BatchNorm2d(num_filters)
        batch_norm2_0 = nn.BatchNorm2d(num_filters * 2)
        # batch_norm2_1 = nn.BatchNorm2d(num_filters * 2)
        batch_norm4_0 = nn.BatchNorm2d(num_filters * 4)
        # batch_norm4_1 = nn.BatchNorm2d(num_filters * 4)
        batch_norm8_0 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_1 = nn.BatchNorm2d(num_filters * 8)
        batch_norm8_2 = nn.BatchNorm2d(num_filters * 8)
        # batch_norm8_3 = nn.BatchNorm2d(num_filters * 8)
        # batch_norm8_4 = nn.BatchNorm2d(num_filters * 8)
        # batch_norm8_5 = nn.BatchNorm2d(num_filters * 8)
        # batch_norm8_6 = nn.BatchNorm2d(num_filters * 8)
        # batch_norm8_7 = nn.BatchNorm2d(num_filters * 8)

        leaky_relu = nn.LeakyReLU(0.2)
        relu = nn.ReLU()
        tanh = nn.Tanh()

        return nn.Sequential(relu, nn.Upsample(scale_factor=2, mode='bilinear'), 
            dconv1, batch_norm8_0, 	relu, nn.Upsample(scale_factor=2, mode='bilinear'), 
            dconv2, batch_norm8_1, relu,  nn.Upsample(scale_factor=2, mode='bilinear'), 
            dconv3, batch_norm8_2, relu, nn.Upsample(scale_factor=2, mode='bilinear'), 
            dconv4,  batch_norm4_0, relu, nn.Upsample(scale_factor=2, mode='bilinear'), 
            dconv5, batch_norm2_0, relu, nn.Upsample(scale_factor=2, mode='bilinear'), 
            dconv6, batch_norm1_0, relu, nn.Upsample(size=(96,96), mode='bilinear'), 
            dconv7, tanh)
 
class Classifier_W_GRU(nn.Module):
    def __init__(self, options):
        super(Classifier_W_GRU, self).__init__()
        self.gru = gen_GRU()
        self.apply(initialize_weights)

    def forward(self, input):
        batch_size = input.size(0)
        sf, fc, feat = self.gru(input)
        return sf, fc, feat


class NLLSequenceLoss(nn.Module):
    """
    Custom loss function.
    Returns a loss that is the sum of all losses at each time step.
    """
    def __init__(self):
        super(NLLSequenceLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.criterion = nn.NLLLoss()

    def forward(self, input, target):
        loss = 0.0
        logsoftmaxed = self.logsoftmax(input)
        transposed = logsoftmaxed.transpose(0, 1).contiguous()

        for i in range(0, 29):
            loss += self.criterion(transposed[i], target)
        return loss
