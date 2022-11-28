import colorsys
import os

import numpy as np
import paddle
from PIL import ImageDraw, ImageFont
from paddle import nn
import paddle

from model import *
from mydataloader import *
from utils import *
from loss import *


# paddle.set_device("gpu:0")

def preprocess_image(image):
    # mean = [0.40789655, 0.44719303, 0.47026116]
    # std = [0.2886383, 0.27408165, 0.27809834]
    mean = [0.484, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return ((np.float32(image) / 255.) - mean) / std
    
# 预测时使用
class CenterNet(object):
    _defaults = {
        "model_path"        : 'work/r18_stage2_dynamic/Epoch60-Total_Loss0.2503-Val_Loss0.7637',
        "classes_path"      : 'work/model_data/pede.txt',
        "image_size"        : [256,128,3],
        "backbone"          : 'resnet_18', # resnet_18, resnet50_vd, resnet34_vd, resnet101_vd, resnet152_vd
        "pretrained"        : False
        # "confidence"        : 0.3,
        # "nms"               : True,
        # "nms_threhold"      : 0.3,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化centernet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.generate()
        self.n_class = 35
        self.confidence = 0.3
        self.nms_threhold = 0.3
        self.image_size = [256,128,3] # h,w,c
        self.max_per_img = 1
        self.down_ratio = 4
        self.regress_ltrb = True
        self.for_mot = False
        self.model_path = kwargs['model_path']

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        #----------------------------------------#
        #   计算种类数量
        #----------------------------------------#
        self.num_classes = len(self.class_names)

        #----------------------------------------#
        #   创建centernet模型
        #----------------------------------------#
        self.centernet = CenterNet_ResNet(num_classes=1, pretrain=False, config=self.__dict__)  # 动态图加载方式
        state_dict = paddle.load(self.model_path + '.pdparams')
        self.centernet.set_state_dict(state_dict)
        # self.centernet = paddle.jit.load(self.model_path)  # 静态图模型直接加载

        self.centernet.eval()

        print('{} model, and classes loaded.'.format(self.model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))


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


    def call(self, hm, wh, reg, im_shape, scale_factor=[1.2,1.2]):
        #print(hm.shape, wh.shape, reg.shape)
        heat = self._simple_nms(hm)
        scores, inds, topk_clses, ys, xs = self._topk(heat)
        scores = scores.unsqueeze(1)
        clses = topk_clses.unsqueeze(1)

        reg_t = paddle.transpose(reg, [0, 2, 3, 1])
        # Like TTFBox, batch size is 1.
        # TODO: support batch size > 1
        reg = paddle.reshape(reg_t, [-1, reg_t.shape[-1]])
        reg = paddle.gather(reg, inds)
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

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, image_id = None, record = False):
        image_shape = np.array(np.shape(image)[0:2]) # hwc
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = image.resize((self.image_size[1], self.image_size[0])) 
        photo = np.array(crop_img, np.float32)
        #----------------------------------------------------------------------------------#
        #   将RGB转化成BGR，这是因为原始的centernet_hourglass权值是使用BGR通道的图片训练的
        #----------------------------------------------------------------------------------#
        #photo = np.array(crop_img,dtype = np.float32)[:,:,::-1]
        #-----------------------------------------------------------#
        #   图片预处理，归一化。获得的photo的shape为[1, 512, 512, 3]
        #-----------------------------------------------------------#
        photo = np.reshape(np.transpose(preprocess_image(photo), (2, 0, 1)), [1, self.image_size[2], self.image_size[0], self.image_size[1]])
        
        with paddle.no_grad():
            images = paddle.to_tensor(np.asarray(photo), dtype='float32')

            outputs = self.centernet(images)
            # print(outputs[-1])
            
            output = outputs[0]
            # print(output.shape)
            if len(output)<=0:
                return image
                
            bboxes, scores = self.call(output[0], output[1], output[2], self.image_size)

            left, top, right, bottom = bboxes[0]
            boxes = resize_centernet_correct_boxes(top,left,bottom,right,np.array([self.image_size[0],self.image_size[1]]),image_shape)
        
        if record:
            if not os.path.exists("work/test_det_results/"):
                os.makedirs("work/test_det_results/")

        top, left, bottom, right = boxes
        print(scores)
        if record and scores >= self.confidence:
            print(image_id)
            predicted_class = 0
            f_r = open("work/test_det_results/%s.txt"%image_id, "a")
            f_r.write("%s %s %s %s %s %s\n" % (predicted_class, scores.numpy(), str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        return image

# 执行预测

import os
import argparse
from tqdm import tqdm
from PIL import Image
# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str)
parser.add_argument('--gt_file', type=str)
parser.add_argument('--model_file', type=str)
args = parser.parse_args()

centernet = CenterNet(model_path=args.model_file)

record = True  # 是否记录检测结果至txt文件

test_img_path = args.img_dir
test_img_list = []

f = open(args.gt_file, 'r')

for line in f.readlines():
    line = line.strip()
    items = line.split(' ')
    test_img_list.append(items[0])

img_list_sorted = sorted(test_img_list)
# print(img_list_sorted)

for img in tqdm(img_list_sorted):
    image = Image.open(img)
    img = img.split('/')[-1]
    image_id = img.split(".")[0]
    r_image = centernet.detect_image(image, image_id, record)
