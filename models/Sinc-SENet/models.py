"""
    Author: Knoxliu (dengkailiu@whu.edu.cn)
    All rights reserved.
"""
from torch import nn
from Sincnet import SincConv_fast
from senet import SEBasicBlock
import torch
import torch.nn.functional as F
import numpy as np
from tcn import TemporalConvNet as TCN
import sys

def conv1x3(inplanes, outplanes, kernel_size=3, stride=1, padding=1):
    ''' 1x3 convolution '''
    return nn.Conv1d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding)


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        self.bn = nn.BatchNorm1d(out_channels*2)
        self.type = type
        if type:
            self.filter = nn.Conv1d(in_channels, 2*out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, out_channels*2)

    def forward(self, x):
        x = self.filter(x)
        if self.type:
            x = self.bn(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class ResNetBlock(nn.Module):
    def __init__(self, in_depth, depth, first=False):
        super(ResNetBlock, self).__init__()
        self.first = first
        self.conv1 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(depth)
        self.lrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=3, padding=1)
        self.conv11 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=3, padding=1)
        if not self.first :
            self.pre_bn = nn.BatchNorm2d(in_depth)

    def forward(self, x):
        # x is (B x d_in x T)
        prev = x
        prev_mp =  self.conv11(x)
        if not self.first:
            out = self.pre_bn(x)
            out = self.lrelu(out)
        else:
            out = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        # out is (B x depth x T/3)
        out = out + prev_mp
        return out

class ResBasicBlock(nn.Module):
    def __init__(self, planes):
        super(ResBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(planes)
        self.re1 = nn.ReLU(inplace=True)
        self.cnn1 = conv1x3(planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.re2 = nn.ReLU(inplace=True)
        self.cnn2 = conv1x3(planes)

    def forward(self, x):
        residual = x
        x = self.cnn2(self.re2(self.bn2(self.cnn1(self.re1(self.bn1(x))))))
        x += residual
        return x


class SElayer(nn.Module):
    def __init__(self, channel, reduction=16, dimension=1):
        super(SElayer, self).__init__()
        self.dimension = dimension
        if self.dimension == 1:
            self.avg_pool = nn.AdaptiveAvgPool1d(1) # F_squeeze
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1) # F_squeeze
        self.fc = nn.Sequential(
            # nn.Linear(channel, channel//reducion),
            # nn.Linear(channel//reducion, channel),
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel//reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.dimension == 1:
            b, c, _,  = x.size()
        else:
            b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        print("\nb: {}, c: {}\n".format(b, c))

        if self.dimension == 1:
            y = self.fc(y).view(b, c, 1)
        else:
            y = self.fc(y).view(b, c, 1, 1)

        f = open("./outputs/printing_y.txt", "a+")
        print("======printing y: ======\n", file=f)
        torch.set_printoptions(profile="full")
        print("x: ", x.size(), file=f)
        print("y: ", y.size(), file=f)
        print("y: ", y, file=f)
        torch.set_printoptions(profile="default")
        print("\n======printing over.======\n", file=f)
        f.close()

        return x * y.expand_as(x)


class SEBlock(nn.Module):
    def __init__(self, planes, kernel_size=3, stride=1, padding=1):
        super(SEBlock, self).__init__()
        self.bn = nn.BatchNorm1d(planes)
        # self.conv1 = nn.Conv1d(planes, planes, kernel_size=5)
        self.conv1 = conv1x3(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = conv1x3(planes, planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.se = SElayer(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)
        return out




class MFCCModel(nn.Module):
    def __init__(self):
        super(MFCCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(32, 32,  True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = ResNetBlock(32, 32,  False)
        self.block3 = ResNetBlock(32, 32,  False)
        self.block4 = ResNetBlock(32, 32, False)
        self.block5 = ResNetBlock(32, 32, False)
        self.block6 = ResNetBlock(32, 32, False)
        self.block7 = ResNetBlock(32, 32, False)
        self.block8 = ResNetBlock(32, 32, False)
        self.block9 = ResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(480, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.mp(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


class SpectrogramModel(nn.Module):
    def __init__(self):
        super(SpectrogramModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(32, 32,  True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = ResNetBlock(32, 32,  False)
        self.block3 = ResNetBlock(32, 32,  False)
        self.block4= ResNetBlock(32, 32, False)
        self.block5= ResNetBlock(32, 32, False)
        self.block6 = ResNetBlock(32, 32, False)
        self.block7 = ResNetBlock(32, 32, False)
        self.block8 = ResNetBlock(32, 32, False)
        self.block9 = ResNetBlock(32, 32, False)
        self.block10 = ResNetBlock(32, 32, False)
        self.block11 = ResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        #out = self.block2(out)
        #out = self.mp(out)
        out = self.block3(out)
        #out = self.block4(out)
        #out = self.mp(out)
        out = self.block5(out)
        #out = self.block6(out)
        #out = self.mp(out)
        out = self.block7(out)
        #out = self.block8(out)
        #out = self.mp(out)
        out = self.block9(out)
        #out = self.block10(out)
        #out = self.mp(out)
        out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        #out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


class CQCCModel(nn.Module):
    def __init__(self):
        super(CQCCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(32, 32, True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = ResNetBlock(32, 32, False)
        self.block3 = ResNetBlock(32, 32, False)
        self.block4 = ResNetBlock(32, 32, False)
        self.block5 = ResNetBlock(32, 32, False)
        self.block6 = ResNetBlock(32, 32, False)
        self.block7 = ResNetBlock(32, 32, False)
        self.block8 = ResNetBlock(32, 32, False)
        self.block9 = ResNetBlock(32, 32, False)
        self.block10 = ResNetBlock(32, 32, False)
        self.block11 = ResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        # out = self.block2(out)
        # out = self.mp(out)
        out = self.block3(out)
        # out = self.block4(out)
        out = self.mp(out)
        out = self.block5(out)
        # out = self.block6(out)layers
        out = self.mp(out)
        out = self.block7(out)
        # out = self.block8(out)
        out = self.mp(out)
        out = self.block9(out)
        # out = self.block10(out)
        out = self.mp(out)
        out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out

class Sincnetmodelv1(nn.Module):
    def __init__(self, block=SEBlock, layer=[4], CNN_N_filt=80, CNN_len_filt=251, fs=16000, num_class=2, focal_loss=None):
        self.expansion = 2
        super(Sincnetmodelv1, self).__init__()
        # inplanes--->the number of output channel in sincConv
        self.inplanes = CNN_N_filt
        self.fs = fs
        self.focal_loss = focal_loss
        self.relu = nn.ReLU()
        self.sincConv = SincConv_fast(self.inplanes, CNN_len_filt, self.fs)
        self.bn = nn.BatchNorm1d(self.inplanes)
        self.se = SElayer(self.inplanes)
        self.layer = []
        self.conv1 = self.make_layer(block, self.inplanes, layer[0])
        # self.conv12 = nn.Sequential(nn.Conv1d(self.inplanes, self.inplanes*self.expansion, 3),
        #                              nn.BatchNorm1d(self.inplanes*self.expansion))
        # self.conv2 = self.make_layer(block, self.inplanes*self.expansion, layer[1])

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.inplanes*len(layer), num_class)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def make_layer(self, block, planes, nblocks, kernel_size =3,
                   stride=1, padding=1):
        layers = []
        for i in range(nblocks):
            # layers.append(nn.AvgPool1d(kernel_size=kernel_size, stride=stride))
            layers.append(block(planes, kernel_size=kernel_size, stride=stride,
                            padding=padding))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.sincConv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)
        out = self.maxpool(out)

        out = self.conv1(out)
        out = self.dropout(out)
        # out = self.conv12(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.dropout(out)

        out = self.avgpool(out).view(x.size()[0], -1)
        out = self.fc(out)
        if self.focal_loss: return out
        else: return F.log_softmax(out, dim=-1)

class Sincnetmodelv2(nn.Module):
    def __init__(self, fs=16000, s_len=0.015, hopping=0.005,
                 tConv_kernel_num = 60, tConv_kernel_size = 160, fConv_kernel_num = 128,
                 fConv_kernel_size = 8, lstm_num_units=256, lstm_num_layer=2, num_classes=2):
        super(Sincnetmodelv2, self).__init__()
        w_len = int(fs * s_len)
        out_channel = [128]*8
        # hop_len = int(fs * hopping)
        # N_frame = (time_interval - w_len) // hop_len + 1

        # normal conv
        # self.tConv1 = nn.Conv2d(1, tConv_kernel_num, (1, tConv_kernel_size))
        # self.tmax_pl1 = nn.MaxPool2d((1, w_len - tConv_kernel_size + 1), stride=(1, 1))

        # sinc conv
        self.tConv2 = SincConv_fast(tConv_kernel_num, tConv_kernel_size, fs, dim=2)
        self.tmax_pl2 = nn.MaxPool2d((1, w_len - (tConv_kernel_size + 1) + 1), stride=(1, 1))

        self.fConv1 = nn.Conv2d(tConv_kernel_num, fConv_kernel_num,
                               (fConv_kernel_size, 1))
        self.se1 = SElayer(tConv_kernel_num, reduction=16, dimension=2)
        #self.se2 = SElayer(fConv_kernel_num, reduction=16, dimension=1)
        self.fmax_pl = nn.MaxPool2d((3, 1), stride=(3, 1))

        self.relu = nn.ReLU()
        # self.tlu = nn.LeakyReLU(1)
        self.bn1 = nn.BatchNorm2d(tConv_kernel_num)
        self.bn2 = nn.BatchNorm2d(fConv_kernel_num)
        self.drop = nn.Dropout(0.5)

        # lstm
        # self.lstm = nn.LSTM(fConv_kernel_num, lstm_num_units, lstm_num_layer)
        # self.fc1 = nn.Linear(lstm_num_units, 256)

        # tcn
        self.tcn = TCN(fConv_kernel_num, out_channel, kernel_size=3)
        self.fc2 = nn.Linear(out_channel[-1], 256)


        self.classifier = nn.Linear(256, num_classes)


    def forward(self, input):
        # print(input.shape, input)

        # normal conv
        # out = self.tConv1(input.unsqueeze(1))
        # out = self.se1(out)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out = self.tmax_pl1(out)
        #out = self.se1(out)

        # sinc conv
        out = self.tConv2(input.unsqueeze(1))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.tmax_pl2(out)
        out = self.se1(out)

        out = self.fConv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fmax_pl(out)
        #out = self.se2(out)

        out = out.squeeze(-1)

        # LSTM
        # out, (h_out, c_out) = self.lstm(torch.transpose(torch.transpose(out, 0, 2), 1, 2))
        # out =self.drop(out)
        # out = self.fc1(out)
        # out =self.drop(out)
        # out = torch.narrow(out, 0, -1, 1).view(out.shape[1], -1)

        # TCN
        out = self.tcn(out)
        out = torch.narrow(out, -1, -1, 1).squeeze(-1)
        out = self.fc2(out)
        out = self.drop(out)

        out = self.classifier(out)
        return F.log_softmax(out, dim=-1)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def act_fun(act_type):
    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!


class sincnet_ori(nn.Module):
    def __init__(self, cnn_N_filt=[80, 60, 60], cnn_len_filt=[251, 5, 5],
                 cnn_max_pool_len=3, fs=16000):
        super(sincnet_ori, self).__init__()
        # sincnet
        self.cnn_N_filt = cnn_N_filt
        self.cnn_len_filt = cnn_len_filt
        self.cnn_max_pool_len = cnn_max_pool_len

        self.cnn_act = 'leaky_relu'
        self.cnn_drop = [0, 0, 0]

        self.cnn_use_laynorm = [True, True, True]
        self.cnn_use_batchnorm = False
        self.cnn_use_laynorm_inp = True
        self.cnn_use_batchnorm_inp = False

        self.input_dim = 64000
        self.fs = fs

        self.N_cnn_lay = len(cnn_N_filt)
        self.conv = nn.ModuleList([])
        self.bn1 = nn.ModuleList([])
        self.ln1 = nn.ModuleList([])
        self.act1 = nn.ModuleList([])
        self.drop1 = nn.ModuleList([])

        if self.cnn_use_laynorm_inp:
            self.ln01 = LayerNorm(self.input_dim)

        # CNN architecture
        current_input = self.input_dim
        for i in range(self.N_cnn_lay):
            N_filt = int(self.cnn_N_filt[i])

            # dropout
            self.drop1.append(nn.Dropout(p=self.cnn_drop[i]))

            # activation
            self.act1.append(act_fun(self.cnn_act))

            # layer norm initialization
            self.ln1.append(
                LayerNorm([N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len)]))

            self.bn1.append(
                nn.BatchNorm1d(N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len),
                               momentum=0.05))

            if i == 0:
                self.conv.append(SincConv_fast(self.cnn_N_filt[0], self.cnn_len_filt[0], self.fs))

            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i - 1], self.cnn_N_filt[i], self.cnn_len_filt[i]))

            current_input = int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len)

        self.out_dim = current_input * N_filt

        # dnn
        self.fc_lay = [512, 512, 512]
        self.fc_drop = [0.5, 0.5, 0.5]
        self.fc_use_laynorm = [True, True, True]
        self.fc_use_batchnorm = False
        self.fc_use_laynorm_inp = True
        self.fc_use_batchnorm_inp = False
        self.fc_act = 'leaky_relu'

        self.wx = nn.ModuleList([])
        self.bn2 = nn.ModuleList([])
        self.ln2 = nn.ModuleList([])
        self.act2 = nn.ModuleList([])
        self.drop2 = nn.ModuleList([])

        # input layer normalization
        if self.fc_use_laynorm_inp:
            self.ln02 = LayerNorm(self.out_dim)

        # input batch normalization
        if self.fc_use_batchnorm_inp:
            self.bn02 = nn.BatchNorm1d([self.out_dim], momentum=0.05)

        self.N_fc_lay = len(self.fc_lay)

        # DNN architecture
        current_input = self.out_dim
        for i in range(self.N_fc_lay):

            # dropout
            self.drop2.append(nn.Dropout(p=self.fc_drop[i]))

            # activation
            self.act2.append(act_fun(self.fc_act))


            # layer norm initialization
            self.ln2.append(LayerNorm(self.fc_lay[i]))
            self.bn2.append(nn.BatchNorm1d(self.fc_lay[i], momentum=0.05))


            # Linear operations
            self.wx.append(nn.Linear(current_input, self.fc_lay[i], bias=False))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(
                torch.Tensor(self.fc_lay[i], current_input).uniform_(-np.sqrt(0.01 / (current_input + self.fc_lay[i])),
                                                                     np.sqrt(0.01 / (current_input + self.fc_lay[i]))))
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.fc_lay[i]))

            current_input = self.fc_lay[i]

        self.fc = nn.Linear(self.fc_lay[-1], 2)


    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[2]
        if bool(self.cnn_use_laynorm_inp):
            x = self.ln01((x))

        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn01((x))

        x = x.view(batch, 1, seq_len)

        for i in range(self.N_cnn_lay):

            if self.cnn_use_laynorm[i]:
                if i == 0:
                    x = self.drop1[i](
                        self.act1[i](self.ln1[i](F.max_pool1d(torch.abs(self.conv[i](x)), self.cnn_max_pool_len))))
                else:
                    x = self.drop1[i](self.act1[i](self.ln1[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len))))

            if self.cnn_use_batchnorm:
                x = self.drop1[i](self.act1[i](self.bn1[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len))))

            if self.cnn_use_batchnorm == False and self.cnn_use_laynorm[i] == False:
                x = self.drop1[i](self.act1[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len)))

        x = x.view(batch, -1)

        # Applying Layer/Batch Norm
        if self.fc_use_laynorm_inp:
            x = self.ln02((x))

        if self.fc_use_batchnorm_inp:
            x = self.bn02((x))

        for i in range(self.N_fc_lay):

            if self.fc_act != 'linear':

                if self.fc_use_laynorm[i]:
                    x = self.drop2[i](self.act2[i](self.ln2[i](self.wx[i](x))))

                if self.fc_use_batchnorm:
                    x = self.drop2[i](self.act2[i](self.bn2[i](self.wx[i](x))))

                if self.fc_use_batchnorm == False and self.fc_use_laynorm[i] == False:
                    x = self.drop2[i](self.act2[i](self.wx[i](x)))

            else:
                if self.fc_use_laynorm[i]:
                    x = self.drop2[i](self.ln2[i](self.wx[i](x)))

                if self.fc_use_batchnorm:
                    x = self.drop2[i](self.bn2[i](self.wx[i](x)))

                if self.fc_use_batchnorm == False and self.fc_use_laynorm[i] == False:
                    x = self.drop2[i](self.wx[i](x))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


class SEResNet(nn.Module):
    """ basic ResNet class: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py """

    def __init__(self, block=SEBasicBlock, layers=[3, 4, 6, 3], num_classes=2,
                 focal_loss=None):

        self.inplanes = 16
        self.focal_loss = focal_loss

        super(SEResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x.unsqueeze(1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.size())

        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())
        x = self.layer4(x)
        # print(x.size())

        x = self.avgpool(x).view(x.size()[0], -1)
        # print(x.shape)
        out = self.classifier(x)

        # print(out.shape)

        if self.focal_loss:
            return out
        else:
            return F.log_softmax(out, dim=-1)


class CLDNN(nn.Module):
    def __init__(self, fs=16000, s_len=0.035, hopping=0.01,
                 tConv_kernel_num = 39, tConv_kernel_size = 400, fConv_kernel_num = 256,
                 fConv_kernel_size = 8, lstm_num_units=256, lstm_num_layer=2, num_classes=2):
        super(CLDNN, self).__init__()
        w_len = int(fs * s_len)
        # hop_len = int(fs * hopping)
        # N_frame = (time_interval - w_len) // hop_len + 1
        self.tmax_pl = nn.MaxPool2d((1, w_len - tConv_kernel_size + 1), stride=(1, 1))
        self.tConv = nn.Conv2d(1, tConv_kernel_num, (1, tConv_kernel_size))
        self.fConv = nn.Conv2d(tConv_kernel_num, fConv_kernel_num,
                               (fConv_kernel_size, 1))
        self.fmax_pl = nn.MaxPool2d((3, 1), stride=(3, 1))
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(tConv_kernel_num)
        self.bn2 = nn.BatchNorm2d(fConv_kernel_num)
        self.drop = nn.Dropout(0.5)
        # self.h_0 = torch.randn(1, batch_size, self.lstm_num_units)
        # self.c_0 = torch.randn(1, batch_size, self.lstm_num_units)
        # seq_len = (N_frame - self.fConv_kernel_size + 1) // 3
        self.lstm1 = nn.LSTM(fConv_kernel_num, lstm_num_units, lstm_num_layer)
        self.fc = nn.Linear(lstm_num_units, 512)
        self.classifier = nn.Linear(512, num_classes)


    def forward(self, input):
        # print(input.shape, input)
        out = self.tConv(input.unsqueeze(1))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.tmax_pl(out)
        # out = torch.log(out)
        out = self.fConv(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fmax_pl(out)

        out = out.squeeze(-1)

        out, (h_out, c_out) = self.lstm1(torch.transpose(torch.transpose(out, 0, 2), 1, 2))
        out =self.drop(out)
        out = self.fc(out)
        out =self.drop(out)
        # print(out)
        out = torch.narrow(out, 0, -1, 1).view(out.shape[1], -1)
        out = self.classifier(out)

        return F.log_softmax(out, dim=-1)





