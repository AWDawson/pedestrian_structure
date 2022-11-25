import os

gt = './work/cls_gt.txt'
pred = './work/cls_pred.txt'

attris = ['direction', 'gender', 'age', 'color_top', 'color_bottom']
attris_idx = {'direction':[0,4], 'gender':[4,7-1], 'age':[7,11-1], 'color_top':[11+1,23],
              'color_bottom':[23+1,35]}

error = 0
for i in range(len(attris)):
    attri = attris[i]
    attri_idx = attris_idx[attri]
    f = open(gt, 'r')
    gt_dic = {}
    for line in f.readlines():
        line = line.strip()
        items = line.split(' ')
        if items[-1] == attri:
            gt_dic[items[0]] = items[1]

    f = open(pred, 'r')
    pred_dic = {}
    for line in f.readlines():
        line = line.strip()
        items = line.split(' ')
        pred_dic[items[0]] = items[1]
    
    total = 0
    yes = 0
    #print(len(gt_dic), len(pred_dic))
    for k in gt_dic:
        if pred_dic.get(k, -1) == -1: 
            error += 1
            continue
        # print(k, gt_dic[k], pred_dic[k])
        gt_ = '$'.join(gt_dic[k].split(',')[attri_idx[0]:attri_idx[1]])
        pred_ = '$'.join(pred_dic[k].split(',')[attri_idx[0]:attri_idx[1]])
        if gt_ == pred_:
            yes += 1
        total += 1
    print(attris[i], 'acc:', yes/total)

#print(error)
