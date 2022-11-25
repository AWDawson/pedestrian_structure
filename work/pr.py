import os
from collections import defaultdict

def get_gts(anno_file):
    """
    get gt annos.
    """
    annos = defaultdict(list)
    with open(anno_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.strip().split()
            img_name = words[0]
            l, u, r, d = words[1].split(',')
            annos[img_name] = [int(l), int(u), int(r), int(d)]
    return annos

def get_recs(anno_file):
    annos = {}
    with open(anno_file, 'r') as f:
        for line in f.readlines():
            if len(line.strip().split(' ')) < 2: continue
            words = line.strip().split()
            img_name = words[0]
            l, u, r, d = words[1].split(',')
            annos[img_name] = [int(l), int(u), int(r), int(d)]
    return annos

def IOU(rec1, rec2):
    """
    computing IoU
    rec1: (x0, y0, x1, y1)
    rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangle
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    #print(top_line, left_line, right_line, bottom_line)

    # judge if there is an intersect area
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

if __name__ == '__main__':
    gt_file = './work/gt_6k.txt'
    result_file = './work/pred.txt'

    recs = get_recs(result_file)
    gts = get_gts(gt_file)
    #print(len(recs), len(gts))

    tp, fp, fn, total = 0, 0, 0, 0
    filtered = 0
    cnts = 0
    for img_name in gts.keys():
        cnts += 1
        items = gts[img_name]
        
        if img_name not in recs.keys():
            #print(cnts, img_name)
            fn += 1
        else:
            #print(cnts, img_name)
            rec = recs[img_name]

            if IOU(items, rec) > 0.5:
                tp += 1
            else:
                fp += 1
                fn += 1
    #    total += 1
    #print(tp, fp, fn)
    print('head p:', tp/(tp+fp), 'head r:', tp/(tp+fn))
    #print('p={}, r={}'.format(tp/(total - filtered), (total - filtered - fn)/total))
