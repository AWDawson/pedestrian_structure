import os

files = '../dataset/test_0617_QAv2_merge_age_head.txt'
s = ""

name_dic = {}

f = open(files, 'r')
for line in f.readlines():
    line = line.strip()
    items = line.split(' ')
    #print(items)
    name, label, _ = items
    name = name.split('/')[-1][:-4]
    if name_dic.get(name, -1) == -1:
        name_dic[name] = 1
    else: continue
    l, u, r, d, _ = label.split(',')
    if l == '-1': continue
    #txt = os.path.join(f'test_results/{file}')
    s += f'{name} {l},{u},{r},{d}\n'
    #break
with open('gt.txt', 'w') as f:
    f.write(s)
