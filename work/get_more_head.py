import os

all_test = '../dataset/test_data.txt'
dic = {}
f = open(all_test, 'r')
for line in f.readlines():
    line = line.strip()
    items = line.split(' ')
    dic[items[0]] = 1

all_label = '../dataset/test_10_merge_age.txt'
f = open(all_label, 'r')
s = ''
for line in f.readlines():
    line = line.strip()
    items = line.split(' ')
    name = items[0].split('/')[-1]
    if dic.get(name, -1) != -1:
        bbox = items[1][:-2]
        if bbox != '-1,-1,-1,-1':
            name = name.split('.')[0]
            s += f'{name} {bbox}\n'
with open('gt_6k.txt', 'w') as f:
    f.write(s)