import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from nets.mbv3 import mobilenetv3
from nets.mobilenetv2 import mobilenetv2
from nets.mbv2_ca import mbv2_ca
from nets.resnet import resnet50


class Resnet(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(pretrained)

        # --------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,1024和30,30,2048
        # --------------------------------------------------------------------------------------------#
        if dilate_scale == 8:
            model.layer3.apply(partial(self._nostride_dilate, dilate=2))
            model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate, dilate=2))

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu1 = model.relu1
        self.conv2 = model.conv2
        self.bn2 = model.bn2
        self.relu2 = model.relu2
        self.conv3 = model.conv3
        self.bn3 = model.bn3
        self.relu3 = model.relu3
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        return x_aux, x


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1] #只保留骨干网络

        self.total_idx = len(self.features) #骨干网络的层数 18层
        self.down_idx = [2, 4, 7, 14]

        # --------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,320和30,30,96
        # --------------------------------------------------------------------------------------------#
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x_1 = self.features[:4](x)  #[bs, 24, 119, 119]
        x_2 = self.features[:7](x)  #[bs, 32, 60, 60]
        x_aux = self.features[:14](x)   #[bs, 96, 30, 30]
        x = self.features[14:](x_aux)   #[bs, 320, 30, 30]
        return x_aux, x, x_1, x_2

class MobileNetV2_CA(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=False):
        super(MobileNetV2_CA, self).__init__()
        from functools import partial

        model = mbv2_ca(pretrained=pretrained)
        self.features = model.features #只保留骨干网络

        self.total_idx = len(self.features) #骨干网络的层数 18层
        self.down_idx = [2, 4, 7, 14]

        # --------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,320和30,30,96
        # --------------------------------------------------------------------------------------------#
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x_1 = self.features[:4](x)  #[bs, 24, 119, 119]
        x_2 = self.features[:7](x)  #[bs, 32, 60, 60]
        x_aux = self.features[:14](x)   #[bs, 96, 30, 30]
        x = self.features[14:](x_aux)   #[bs, 320, 30, 30]
        return x_aux, x, x_1, x_2

#有预训练文件
class MobV3(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobV3, self).__init__()
        from functools import partial
        model = mobilenetv3(downsample_factor=downsample_factor, pretrained=pretrained)
        self.features = model.features
        self.total_idx = len(self.features)    #16层 骨干网络
        self.down_idx = [8, 14]

        # --------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,320和30,30,96
        # --------------------------------------------------------------------------------------------#
        # if downsample_factor == 8:
        #     for i in range(self.down_idx[-2], self.down_idx[-1]):   #[7,14)
        #         self.features[i].apply(partial(self._nostride_dilate, dilate=2))
        #     # for i in range(self.down_idx[-1], self.total_idx):      #[14, total_idx)
        #     #     self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        # elif downsample_factor == 16:
        #     for i in range(self.down_idx[-1], self.total_idx):
        #         self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        output = self.features(x)
        x_aux = self.features[:14](x)   #[bs, 160, 30, 30]
        # x = self.features[14:](x_aux)   #[bs, 160, 15, 15]
        return x_aux, output


class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)   # 512(resnet50)  or 80(mobilenetv2)
        # -----------------------------------------------------#
        #   分区域进行平均池化
        #   resnet50: 30, 30, 2048 + ... = 30, 30, 4096
        #   mobilenetv2: 30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
        # -----------------------------------------------------#
        self.stages = nn.ModuleList(
            [self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])

        # 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1,
                      bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend(
            [F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

class PSPSkipUpsample(nn.Module):
    def __init__(self, x_channels, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        # 1x1Conv 改变channel
        self.shortcut = nn.Conv2d(x_channels, out_channels, kernel_size=1)


    def forward(self, x, up, size):
        #x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=False)

        # x_channels=80, in_channels=64, out_channels=32
        # x1:[80, 30, 30] up1:[32, 60, 60], size1:(60, 60)

        # x_channels=32, in_channels=48, out_channels=24
        # x2:[32, 60, 60] up2:[24, 119, 119], size2:(119, 119)

        # x 经过 1x1 卷积改变channel
        x = self.shortcut(x)        #[80, 30, 30] → [32, 30, 30]
        # x进行上采样到 up 同样的大小
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)#[32, 30, 30] → [32, 60, 60]

        #skip连接 x+up 卷积 [32, 60, 60]+[32, 60, 60] = [64, 60, 60]
        x = self.conv(torch.cat([x, up], 1))    #[64, 60, 60] → [32, 60, 60]

        return x

class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone="resnet50", pretrained=True, aux_branch=True, skip_upsample=True):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone == "resnet50":
            self.backbone = Resnet(downsample_factor, pretrained)
            aux_channel = 1024
            out_channel = 2048
        elif backbone == "mobilenet":
            # ----------------------------------#
            #   获得两个特征层
            #   f4为辅助分支    [30,30,96]
            #   o为主干部分     [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2(downsample_factor, pretrained=pretrained)
            aux_channel = 96
            out_channel = 320
        elif backbone == "mobilenet_ca":
            # ----------------------------------#
            #   获得两个特征层
            #   f4为辅助分支    [30,30,96]
            #   o为主干部分     [30,30,320]
            # ----------------------------------#
            self.backbone = MobileNetV2_CA(downsample_factor, pretrained=pretrained)
            aux_channel = 96
            out_channel = 320
        elif backbone == "mbv3":
            self.backbone = MobV3(downsample_factor)
            aux_channel = 96
            out_channel = 160
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

        # --------------------------------------------------------------#
        #	PSP模块，分区域进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   30,30,320 -> 30,30,80 -> 30,30,21
        # --------------------------------------------------------------#
        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),    #[bs, 160 or 320, 60, 60]
            #分类
            # nn.Conv2d(out_channel // 4, num_classes, kernel_size=1)
        )

        #### 增加skip连接和上采样操作 ####
        self.skip_upsample = skip_upsample
        if self.skip_upsample:
            if backbone == "mobilenet" or backbone=="mobilenet_ca":
                self.up_1 = PSPSkipUpsample(80, 64, 32)
                self.up_2 = PSPSkipUpsample(32, 48, 24)
                self.classifier = nn.Conv2d(24, num_classes, kernel_size=1)
        else:
            self.classifier = nn.Conv2d(out_channel // 4, num_classes, kernel_size=1)

        self.aux_branch = aux_branch

        if self.aux_branch:
            # ---------------------------------------------------#
            #	利用特征获得预测结果
            #   30, 30, 96 -> 30, 30, 40 -> 30, 30, 21
            # ---------------------------------------------------#
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel // 8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel // 8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel // 8, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.master_branch)
        if self.skip_upsample:
            self.initialize_weights(self.up_1)
            self.initialize_weights(self.up_2)
        self.initialize_weights(self.classifier)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3]) #[h, w]
        x_aux, x, x_1, x_2 = self.backbone(x) #获得backbone的特征图
        output = self.master_branch(x)  #输入到PSPModule [bs, 80, 30, 30]
        if self.skip_upsample:
            output = self.up_1(output, x_2, (x_2.size()[2], x_2.size()[3]))
            output = self.up_2(output, x_1, (x_1.size()[2], x_1.size()[3]))
        output = self.classifier(output)

        # 最后的特征图进行一次双线性插值上采样，恢复图像大小
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return output   #[bs, classes, h, w]

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()
                elif isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0.0, 0.02)

if __name__ == '__main__':
    model = PSPNet(num_classes=2, backbone='mobilenet_ca', downsample_factor=8, pretrained=True, aux_branch=False, skip_upsample=True).train()
    input = torch.randn(2, 3, 473, 473)
    outputs = model(input)
    print(outputs.size())
    print(model)
    #######查看网络结构#############
    # from torchsummary import summary
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    # summary(model, (3, 473, 473))

    ######网络结构可视化#############
    # from torchviz import make_dot
    # net_struct = make_dot(outputs)
    # net_struct.view()

    #######计算参数量方法2#############
    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #     flops, params = get_model_complexity_info(model, (3, 473, 473), as_strings=True,
    #                                               print_per_layer_stat=True)  # 不用写batch_size大小，默认batch_size=1
    #     print('Flops:  ' + flops)
    #     print('Params: ' + params)

    #######查看训练是否正常#############
    # batch_size, n_classes, h, w = 2, 2, 473, 473
    # import torch.optim as optim
    # import torch.utils
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # model = PSPNet(num_classes=2, backbone='mbv3', downsample_factor=8, pretrained=True,
    #                aux_branch=False).to(device)
    # model.train()
    #
    # input = torch.randn(batch_size, 3, h, w).to(device)
    # label = torch.randn(batch_size, n_classes, h, w).to(device)
    #
    # criterion = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    #
    # for iter in range(50):
    #     optimizer.zero_grad()
    #     output = model(input)
    #     output = torch.sigmoid(output)
    #     loss = criterion(output, label)
    #     loss.backward()
    #     print("iter:{}, loss:{}".format(iter, loss.item()))
    #     optimizer.step()