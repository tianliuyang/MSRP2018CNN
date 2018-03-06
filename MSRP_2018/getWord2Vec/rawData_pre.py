# -*- coding:utf-8 -*-

def rawdata_pre(path):
    file = open(path,'r',encoding='utf-8')
    lines = file.readlines()
    label_sen = []
    for line in lines:
        info = line.split('\t')
        label_sen.append(info[1:])
    return label_sen

path_raw = r'..\RawData\垃圾短信训练集80W条.txt'
label_sen = rawdata_pre(path_raw)
file_sive = open(r'..\RawData\junk_message_80W.txt', 'w', encoding='utf-8')  # 存储 标签—特征文件
for i in range(0, len(label_sen)):
    print(i)
    file_sive.write('\t'.join(label_sen[i]))
file_sive.close()