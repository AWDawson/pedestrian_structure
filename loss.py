# 定义损失函数
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.nn.functional as F

import numpy as np


def focal_loss(pred, target):
    pred = paddle.transpose(pred, [0,2,3,1])
    # pred = pred.permute(0,2,3,1)

    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    # pos_inds = target.equal(1).float()
    pos_inds = target == 1
    pos_inds = pos_inds.astype('float32')
    # neg_inds = target.lt(1).float()
    neg_inds = target < 1
    neg_inds = neg_inds.astype('float32')
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = paddle.pow(1 - target, 4)
    
    pred = paddle.clip(pred, 1e-6, 1 - 1e-6)
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = paddle.log(pred) * paddle.pow(1 - pred, 2) * pos_inds
    neg_loss = paddle.log(1 - pred) * paddle.pow(pred, 2) * neg_weights * neg_inds
    
    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    num_pos = pos_inds.astype('float32').sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def reg_l1_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    # pred = pred.permute(0,2,3,1)
    pred = paddle.transpose(pred, [0,2,3,1])
    # expand_mask = paddle.unsqueeze(mask,-1).repeat(1,1,1,2)
    expand_mask = paddle.tile(paddle.unsqueeze(mask,-1), [1,1,1,2])

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss


def BCELoss(x, label):
    """
    Binary Cross entropy loss
    """
    #sigmoid = paddle.nn.Sigmoid()
    #print(x, label)
    # w = np.array([1 ,1 ,1, 1, 2.0, 2.0,   0.1, 1, 1, 1,   0.1 ,  0.1,1,1,1,1,1,1,1,1,1,1,1,   0.1,1,1,1,1,1,1,1,1,1,1,1])
    # w = np.array([1 ,1 ,1, 1, 5.0, 5.0,   0.1, 1, 1, 1,   0.1 ,  0.1,1,1,1,1,1,1,1,1,1,1,1,   0.1,1,1,1,1,1,1,1,1,1,1,1])
    # w = np.array([0.1 ,0.1 ,0.1, 0.1, 5.0, 5.0, 0.1, 0.1, 0.1, 0.1, 0.1 ,1,1,1,1,1,1,1,1,1,1,1,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    # w = paddle.to_tensor(w, dtype='float32')
    gt = paddle.to_tensor(label)
    # loss = F.binary_cross_entropy(x.astype('float32'), gt.astype('float32'), weight=w)
    loss = F.binary_cross_entropy(x.astype('float32'), gt.astype('float32'))
    return loss


def attri_focal_loss(logit, label, class_dim, gamma=2.0, alpha=0.25, smooth=None):
    """Calculate focal loss
    Returns:
        weighted focal loss
    """
    label_int = label.astype('int')
    # print(label_int==1)
    with paddle.no_grad():
        # alpha_ = paddle.full_like(logit, 1-alpha)
        alpha_ = np.zeros(logit.shape)
        alpha_[:,:] = 1-alpha
        # print(alpha_)
        alpha_[label_int.numpy() == 1] = alpha
    alpha_ = paddle.to_tensor(alpha_)
    pt = paddle.where(label_int==1, logit, 1-logit)
    gt = paddle.to_tensor(label)
    bce_loss = F.binary_cross_entropy(logit.astype('float32'), gt.astype('float32'))
    loss = (alpha_ * paddle.pow(1 - pt, gamma) * bce_loss)
    return loss.mean()