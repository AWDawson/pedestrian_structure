import os

files = os.listdir('./work/test_det_results')
s = ""
for file in files:
    txt = os.path.join(f'./work/test_det_results/{file}')
    f = open(txt, 'r')
    for line in f.readlines():
        line = line.strip()
        items = line.split(' ')
        _, _, l, u, r, d = items
        l = max(0, int(l))
        u = max(0, int(u))
        r = max(0, int(r))
        d = max(0, int(d))
        s += f'{file[:-4]} {l},{u},{r},{d}\n'
        break
with open('./work/pred.txt', 'w') as f:
    f.write(s)
