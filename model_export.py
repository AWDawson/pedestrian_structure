from __future__ import absolute_import, division, print_function

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D

from mymodels import resnet_vd, resnet

from utils import *

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
        #   利用ConvTranspose2d进行上采样。
        #   每次特征层的宽高变为原来的两倍。
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
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        #-----------------------------------------------------------------#
        # 热力图预测部分
        self.act = nn.Sigmoid()
        print('num_classes', num_classes)
        self.cls_head = nn.Sequential(
            nn.Conv2D(64, channel,
                      kernel_size=3, padding=1),
            nn.BatchNorm2D(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2D(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        # 宽高预测的部分
        self.wh_head = nn.Sequential(
            nn.Conv2D(64, channel,
                      kernel_size=3, padding=1),
            nn.BatchNorm2D(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2D(channel, 2,
                      kernel_size=1, stride=1, padding=0))

        # 中心点预测的部分
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

    
def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).astype('float32')
    return heat * keep
    

class CenterNet_ResNet(nn.Layer):
    def __init__(self, num_classes = 20, pretrain=False, config=None):
        super(CenterNet_ResNet, self).__init__()
        self.n_class = 35
        self.confidence = 0.15
        self.nms_threhold = 0.3
        self.image_size = [256,128,3] # h,w,c
        self.max_per_img = 1
        self.down_ratio = 4
        self.regress_ltrb = True
        self.for_mot = False
        self.pretrain = pretrain

        if config['backbone'] == 'resnet18':
            self.backbone = resnet.resnet18(pretrained=self.pretrain)
            last_dim = 512
        elif config['backbone'] == 'resnet18_vd':
            self.backbone = resnet_vd.ResNet18_vd(pretrained=self.pretrain, use_ssld=False)
            last_dim = 512
        elif config['backbone'] == 'resnet34_vd':
            self.backbone = resnet_vd.ResNet34_vd(pretrained=self.pretrain, use_ssld=True)
            last_dim = 512
        elif config['backbone'] == 'resnet50_vd':
            self.backbone = resnet_vd.ResNet50_vd(pretrained=self.pretrain, use_ssld=True)
            last_dim = 2048
        elif config['backbone'] == 'resnet101_vd':
            self.backbone = resnet_vd.ResNet101_vd(pretrained=self.pretrain, use_ssld=True)
            last_dim = 2048
        elif config['backbone'] == 'resnet152_vd':
            self.backbone = resnet_vd.ResNet152_vd(pretrained=self.pretrain, use_ssld=False) # paddleclas没有r152的ssld
            last_dim = 2048

        self.decoder = ResNet_Decoder(last_dim)
        #-----------------------------------------------------------------#
        #   对获取到的特征进行上采样，进行分类预测和回归预测
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
        mapp = [4,3,4,12,12]
        for c in range(5):
            self.__setattr__('class_%d' % c, ClassBlock(input_dim=self.avg_pool_channels, class_num=mapp[c], activ='sigmoid') )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    
    def _simple_nms(self, heat, kernel=3):
        """
        Use maxpool to filter the max score, get local peaks.
        """
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
        keep = paddle.cast(hmax == heat, 'float32')
        return heat * keep

    def _topk(self, scores):
        """
        Select top k scores and decode to get xy coordinates.
        """
        k = self.max_per_img
        shape_fm = paddle.shape(scores)
        shape_fm.stop_gradient = True
        cat, height, width = shape_fm[1], shape_fm[2], shape_fm[3]
        # batch size is 1
        scores_r = paddle.reshape(scores, [cat, -1])
        topk_scores, topk_inds = paddle.topk(scores_r, k)
        topk_scores, topk_inds = paddle.topk(scores_r, k)
        topk_ys = topk_inds // width
        topk_xs = topk_inds % width

        topk_score_r = paddle.reshape(topk_scores, [-1])
        topk_score, topk_ind = paddle.topk(topk_score_r, k)
        k_t = paddle.full(paddle.shape(topk_ind), k, dtype='int64')
        topk_clses = paddle.cast(paddle.floor_divide(topk_ind, k_t), 'float32')

        topk_inds = paddle.reshape(topk_inds, [-1])
        topk_ys = paddle.reshape(topk_ys, [-1, 1])
        topk_xs = paddle.reshape(topk_xs, [-1, 1])
        topk_inds = paddle.gather(topk_inds, topk_ind)
        topk_ys = paddle.gather(topk_ys, topk_ind)
        topk_xs = paddle.gather(topk_xs, topk_ind)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    
    def forward(self, x):
        feat = self.backbone(x)
        x = self.avg_pool(feat)
        x = self.flatten(x)
        pred_label = [self.__getattr__('class_%d' % c)(x) for c in range(5)]
        pred_label = paddle.concat(pred_label, 1)
        
        outputs = self.head(self.decoder(feat))
        bboxes, scores = self.call(outputs[0], outputs[1], outputs[2], self.image_size)
        # bboxes = paddle.to_tensor(np.array([[1.1,1.1,2.2,2.2]]))
        # scores = paddle.to_tensor(np.array([[0.5]]))
        print('bboxes.shape', bboxes.shape)
        print('scores.shape', scores.shape)
        print(pred_label.shape)
        direction = pred_label[:,:4]
        gender = pred_label[:,4:7]
        age = pred_label[:,7:11]
        color_up = pred_label[:,11:23]
        color_down = pred_label[:,23:]
        
        # direction = pred_label
        # gender = pred_label
        # age = pred_label
        # color_up = pred_label
        # color_down = pred_label
        
        return bboxes, scores, direction.argmax(), gender.argmax(), age.argmax(), color_up.argmax(), color_down.argmax()
    
        
    def call(self, hm, wh, reg, im_shape, scale_factor=[1.2,1.2]):
        #print(hm.shape, wh.shape, reg.shape)
        heat = self._simple_nms(hm)
        scores, inds, topk_clses, ys, xs = self._topk(heat)
        scores = scores.unsqueeze(1)
        clses = topk_clses.unsqueeze(1)

        reg_t = paddle.transpose(reg, [0, 2, 3, 1])
        #print(reg_t.shape)
        # Like TTFBox, batch size is 1.
        # TODO: support batch size > 1
        reg = paddle.reshape(reg_t, [-1, reg_t.shape[-1]])
        reg = paddle.gather(reg, inds)
        #print(reg.shape)
        xs = paddle.cast(xs, 'float32')
        ys = paddle.cast(ys, 'float32')
        xs = xs + reg[:, 0:1]
        ys = ys + reg[:, 1:2]

        wh_t = paddle.transpose(wh, [0, 2, 3, 1])
        wh = paddle.reshape(wh_t, [-1, wh_t.shape[-1]])
        wh = paddle.gather(wh, inds)

        wh01 = wh[:, 0:1]
        wh12 = wh[:, 1:2]
        wh23 = wh[:, 0:1]
        wh34 = wh[:, 1:2]
        if self.regress_ltrb:
            x1 = xs - wh01 / 2
            y1 = ys - wh12 / 2
            x2 = xs + wh23 / 2
            y2 = ys + wh34 / 2

        n, c, feat_h, feat_w = paddle.shape(hm)
        padw = (feat_w * self.down_ratio - im_shape[1]) / 2
        padh = (feat_h * self.down_ratio - im_shape[0]) / 2
        x1 = x1 * self.down_ratio
        y1 = y1 * self.down_ratio
        x2 = x2 * self.down_ratio
        y2 = y2 * self.down_ratio

        x1 = x1 - padw
        y1 = y1 - padh
        x2 = x2 - padw
        y2 = y2 - padh

        bboxes = paddle.concat([x1, y1, x2, y2], axis=1)
        return bboxes, scores
        # scale_factor = paddle.to_tensor(scale_factor)
        # scale_factor = paddle.reshape(scale_factor, shape=[-1, 2])
        # scale_y = scale_factor[:,0:1]
        # scale_x = scale_factor[:,1:2]
        # scale_expand = paddle.concat(
        #     [scale_x, scale_y, scale_x, scale_y], axis=1)
        # boxes_shape = bboxes.shape[:]
        # scale_expand = paddle.expand(scale_expand, shape=boxes_shape)
        # bboxes = paddle.divide(bboxes, scale_expand)
        # return bboxes, scores
        # results = paddle.concat([clses, scores, bboxes], axis=1)
        # if self.for_mot:
        #     return results, inds, topk_clses
        # else:
        #     return results, paddle.shape(results)[0:1], topk_clses
        
#         pred_hms, pred_whs, pred_offsets = outputs[0],outputs[1],outputs[2]
#         pred_hms = pool_nms(pred_hms)
#         b, c, output_h, output_w = pred_hms.shape
#         if b == -1: b = 1
#         detects = []
#         batch = 0
#         heat_map    = paddle.transpose(pred_hms[batch], [1,2,0]).reshape([-1,c])
#         # pred_wh     = pred_whs[batch].permute(1,2,0).reshape([-1,2])
#         pred_wh    = paddle.transpose(pred_whs[batch], [1,2,0]).reshape([-1,2])
#         # pred_offset = pred_offsets[batch].permute(1,2,0).reshape([-1,2])
#         pred_offset    = paddle.transpose(pred_offsets[batch], [1,2,0]).reshape([-1,2])
        
#         yv, xv = paddle.meshgrid(paddle.arange(0, output_h), paddle.arange(0, output_w))
#         xv, yv = xv.flatten().astype('float32'), yv.flatten().astype('float32')
        
#         class_conf = paddle.max(heat_map, axis=-1)
#         class_pred = paddle.argmax(heat_map, axis=-1)
#         mask = class_conf > self.confidence
        
#         mask_indexes = layers.where(mask).numpy()#.squeeze()
#         print(mask_indexes)
#         # mask_indexes = list(mask_indexes)
#         # if len(mask_indexes) > 50:
#         #     mask_indexes = mask_indexes[:50]
#         pred_wh_mask = []
#         for i in mask_indexes:
#             pred_wh_mask.append(pred_wh[int(i):int(i+1), :])
#         pred_wh_mask = paddle.concat(pred_wh_mask)

#         pred_offset_mask = []
#         for i in mask_indexes:
#             pred_offset_mask.append(pred_offset[int(i):int(i+1), :])
#         pred_offset_mask = paddle.concat(pred_offset_mask)
#         if len(pred_wh_mask)==0:
#             detects.append([])

#         #----------------------------------------#
#         #   计算调整后预测框的中心
#         #----------------------------------------#
#         xv_mask = []
#         for i in mask_indexes:
#             xv_mask.append(xv[int(i)])
#         xv_mask = paddle.concat(xv_mask)

#         yv_mask = []
#         for i in mask_indexes:
#             yv_mask.append(yv[int(i)])
#         yv_mask = paddle.concat(yv_mask)

#         xv_mask = paddle.unsqueeze(xv_mask + pred_offset_mask[:, 0], -1)
#         yv_mask = paddle.unsqueeze(yv_mask + pred_offset_mask[:, 1], -1)
#         #----------------------------------------#
#         #   计算预测框的宽高
#         #----------------------------------------#
#         half_w, half_h = pred_wh_mask[:, 0:1] / 2, pred_wh_mask[:, 1:2] / 2
#         #----------------------------------------#
#         #   获得预测框的左上角和右下角
#         #----------------------------------------#
#         bboxes = paddle.concat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], axis=1)
#         bboxes[:, 0] /= output_w
#         bboxes[:, 1] /= output_h
#         bboxes[:, 2] /= output_w
#         bboxes[:, 3] /= output_h

#         class_conf_mask = []
#         for i in mask_indexes:
#             class_conf_mask.append(class_conf[int(i)])
#         class_conf_mask = paddle.concat(class_conf_mask)

#         class_pred_mask = []
#         for i in mask_indexes:
#             class_pred_mask.append(class_pred[int(i)])
#         class_pred_mask = paddle.concat(class_pred_mask)

#         detect = paddle.concat(
#             [bboxes, paddle.unsqueeze(class_conf_mask,-1), 
#             paddle.unsqueeze(class_pred_mask,-1).astype('float32')], axis=-1
#         )

#         arg_sort = paddle.argsort(detect[:,-2], descending=True)
#         # print(detect.shape) # [50, 6]
#         detect_mask = []
#         for i in arg_sort:
#             detect_mask.append(detect[int(i):int(i)+1, :])
#         detect_mask = paddle.concat(detect_mask)

#         detect = detect_mask
#         detects.append(detect.numpy()[:1])
        
#         return outputs, pred_label
