import os
import numpy as np
import random
import pickle

from easydict import EasyDict
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)



def generate_label(pa100k_data, mode):
    txt_file = '{}.txt'.format(mode)
    label_key = '{}_label'.format(mode)
    image_key = '{}_images_name'.format(mode)
    with open(txt_file, 'w') as fw:
        for i in range(len(pa100k_data[label_key])):
            fname = pa100k_data[image_key][i][0][0]
            labels = pa100k_data[label_key][i]
            labels = [str(x) for x in labels]
            labels_str = ','.join(labels)
            fw.write(fname + ' ' + labels_str + '\n')

def add_pseudo_head(label_file, save_dir, head_dir, mode):
    new_txt = os.path.join(save_dir, '{}_1114.txt'.format(mode))
    fw = open(new_txt, 'w')
    with open(label_file, 'r') as f:
        for line in f.readlines():
            img_name, cls_label_str = line.strip().split()
            prefix = img_name[:-4]
            head_txt = prefix + '.txt'
            head_path = os.path.join(head_dir, head_txt)
            if os.path.exists(head_path):
                with open(head_path, 'r') as f_head:
                    head_line = f_head.readlines()[-1]
                    score, x1, y1, x2, y2 = head_line.strip().split()
            else:
                x1, y1, x2, y2 = '-1', '-1', '-1', '-1'
            new_line = img_name + ' ' + ','.join([x1, y1, x2, y2, '0']) + ' ' + cls_label_str + '\n'
            fw.write(new_line)
    fw.close()



def generate_data_description(save_dir, reorder):
    """
    create a dataset description file, which consists of images, labels
    """
    # pa100k_data = loadmat('/mnt/data1/jiajian/dataset/attribute/PA100k/annotation.mat')
    pa100k_data = loadmat(os.path.join(save_dir, 'annotation.mat'))
    print(pa100k_data.keys())
    generate_label(pa100k_data, 'train')
    generate_label(pa100k_data, 'val')
    generate_label(pa100k_data, 'test')
    

if __name__ == "__main__":
    save_dir = './'
    reoder = True
    # generate_data_description(save_dir, reorder=True)
    add_pseudo_head('test.txt', './', './PA100K_head_1107', 'test')
