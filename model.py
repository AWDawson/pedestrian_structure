from __future__ import absolute_import, division, print_function

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D

from mymodels import resnet, resnet_vd


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, stride=stride) # change
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, # change
                    padding=1)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 512,512,3 -> 256,256,64
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        # 256x256x64 -> 128x128x64
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change

        # 128x128x64 -> 128x128x256
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 128x128x256 -> 64x64x512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # 64x64x512 -> 32x32x1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  

        # 32x32x1024 -> 16x16x2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2D(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride),
            nn.BatchNorm2D(planes * block.expansion),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNet_Decoder(nn.Layer):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(ResNet_Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        self.inplanes = inplanes
        
        #----------------------------------------------------------#
        #   16,16,2048 -> 32,32,256 -> 64,64,128 -> 128,128,64
        #   ??????ConvTranspose2d??????????????????
        #   ????????????????????????????????????????????????
        #----------------------------------------------------------#
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]
            layers.append(
                nn.Conv2DTranspose(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0))
            layers.append(nn.BatchNorm2D(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU())
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class ResNet_Head(nn.Layer):
    def __init__(self, num_classes=80, channel=64, bn_momentum=0.1):
        super(ResNet_Head, self).__init__()
        #-----------------------------------------------------------------#
        #   ????????????????????????????????????????????????????????????????????????
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        # ?????????????????????
        self.act = nn.Sigmoid()
        print('num_classes', num_classes)
        self.cls_head = nn.Sequential(
            nn.Conv2D(64, channel,
                      kernel_size=3, padding=1),
            nn.BatchNorm2D(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2D(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        # ?????????????????????
        self.wh_head = nn.Sequential(
            nn.Conv2D(64, channel,
                      kernel_size=3, padding=1),
            nn.BatchNorm2D(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2D(channel, 2,
                      kernel_size=1, stride=1, padding=0))

        # ????????????????????????
        self.reg_head = nn.Sequential(
            nn.Conv2D(64, channel,
                      kernel_size=3, padding=1),
            nn.BatchNorm2D(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2D(channel, 2,
                      kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        hm = self.cls_head(x)
        # print(hm)
        wh = self.wh_head(x)
        offset = self.reg_head(x)
        return self.act(hm), wh, offset


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Layer):
    def __init__(self, input_dim, class_num=1, activ='sigmoid', num_bottleneck=512):
        super(ClassBlock, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]

        add_block = nn.Sequential(*add_block)
        #add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        if activ == 'sigmoid':
            classifier += [nn.Sigmoid()]
        elif activ == 'softmax':
            classifier += [nn.Softmax()]
        elif activ == 'none':
            classifier += []
        else:
            raise AssertionError("Unsupported activation: {}".format(activ))
        classifier = nn.Sequential(*classifier)
        #classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class CenterNet_ResNet(nn.Layer):
    def __init__(self, num_classes=20, pretrain=False, config=None):
        super(CenterNet_ResNet, self).__init__()
        # self.n_class = config['n_class']
        # print(config['backbone'])
        # assert config['backbone'] in ['resnet18', 'resnet50_vd', 'resnet101_vd', 'resnet152_vd'], \
        #         'only support backbone of resnet_18, resnet50_vd, resnet101_vd, resnet152_vd'
        if config['backbone'] == 'resnet_18':
            self.backbone = resnet.resnet18(pretrained=config['pretrained'])
            last_dim = 512
        elif config['backbone'] == 'resnet18_vd':
            self.backbone = resnet_vd.ResNet18_vd(pretrained=True, use_ssld=False)
            last_dim = 512
        elif config['backbone'] == 'resnet34_vd':
            self.backbone = resnet_vd.ResNet34_vd(pretrained=True, use_ssld=True)
            last_dim = 512
        elif config['backbone'] == 'resnet50_vd':
            self.backbone = resnet_vd.ResNet50_vd(pretrained=True, use_ssld=True)
            last_dim = 2048
        elif config['backbone'] == 'resnet101_vd':
            self.backbone = resnet_vd.ResNet101_vd(pretrained=True, use_ssld=True)
            last_dim = 2048
        elif config['backbone'] == 'resnet152_vd':
            self.backbone = resnet_vd.ResNet152_vd(pretrained=True, use_ssld=False) # paddleclas??????r152???ssld
            last_dim = 2048
        else:
            print('only support backbone of resnet_18, resnet18_vd, resnet34_vd, resnet50_vd, resnet101_vd, resnet152_vd')

        self.decoder = ResNet_Decoder(last_dim)
        #-----------------------------------------------------------------#
        #   ????????????????????????????????????????????????????????????????????????
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        self.head = ResNet_Head(channel=64, num_classes=num_classes)

        data_format="NCHW"
        self.avg_pool = AdaptiveAvgPool2D(1, data_format=data_format)
        self.flatten = nn.Flatten()
        self.avg_pool_channels = last_dim
        stdv = 1.0 / math.sqrt(self.avg_pool_channels * 1.0)
        mapp = [1,3,3,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1]
        # mapp = [4,3,4,12,12]
        for c in range(21):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.avg_pool_channels, class_num=mapp[c], activ='sigmoid') )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for c in range(21):
            for param in self.__getattr__('class_%d' % c).parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        for c in range(21):
            for param in self.__getattr__('class_%d' % c).parameters():
                param.requires_grad = True

    def forward(self, x):
        feat = self.backbone(x) # n, c, h, w
        #print(feat.shape)
        x = self.avg_pool(feat)
        x = self.flatten(x)
        pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(21)]
        pred_label = paddle.concat(pred_label, 1)
        return self.head(self.decoder(feat)), pred_label
