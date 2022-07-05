import os
from matplotlib.pyplot import text
import pandas as pd
from tqdm import tqdm
import json
import random
import re
data_dir = 'data/contest_data/train_fine.txt'
data_dir_coarse = "data/contest_data/train_coarse.txt"
'''
这个版本不考虑关键属性的具体粒度，只当做13个标签的多标签分类来做
['图文','版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']
'''
items=[]
class_name=['图文','版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']
class_dict={'图文': ['符合','不符合'],
            '版型': ['修身型', '宽松型', '标准型'],
            '裤型': ['微喇裤', '小脚裤', '哈伦裤', '直筒裤', '阔腿裤', '铅笔裤', 'O型裤', '灯笼裤', '锥形裤', '喇叭裤', '工装裤', '背带裤', '紧身裤'],
            '袖长': ['长袖', '短袖', '七分袖', '五分袖', '无袖', '九分袖'],
            '裙长': ['中长裙', '短裙', '超短裙', '中裙', '长裙'],
            '领型': ['半高领', '高领', '翻领', 'POLO领', '立领', '连帽', '娃娃领', 'V领', '圆领', '西装领', '荷叶领', '围巾领', '棒球领', '方领', '可脱卸帽', '衬衫领', 'U型领', '堆堆领', '一字领', '亨利领', '斜领', '双层领'],
            '裤门襟': ['系带', '松紧', '拉链'],
            '鞋帮高度': ['低帮', '高帮', '中帮'],
            '穿着方式': ['套头', '开衫'],
            '衣长': ['常规款', '中长款', '长款', '短款', '超短款', '超长款'],
            '闭合方式': ['系带', '套脚', '一脚蹬', '松紧带', '魔术贴', '搭扣', '套筒', '拉链'],
            '裤长': ['九分裤', '长裤', '五分裤', '七分裤', '短裤'],
            '类别': ['单肩包', '斜挎包', '双肩包', '手提包']
            }
class_index={'图文': ['0','1'],
            '版型': ['0', '1', '0'],
            '裤型': ['0', '1', '2', '1', '3', '1', '2', '2', '2', '0', '4', '5', '6'],
            '袖长': ['0', '1', '2', '1', '3', '0'],
            '裙长': ['0', '1', '1', '0', '2'],
            '领型': ['0', '0', '1', '1', '0', '2', '1', '3', '4', '5', '1', '6', '7', '1', '2', '1', '8', '9', '10', '11', '12', '13'],
            '裤门襟': ['0', '1', '2'],
            '鞋帮高度': ['0', '1', '1'],
            '穿着方式': ['0', '1'],
            '衣长': ['0', '1', '2', '0', '0', '2'],
            '闭合方式': ['0', '1', '1', '2', '3', '4', '1', '5'],
            '裤长': ['0', '0', '1', '2', '3'],
            '类别': ['0', '1', '2', '3']
            }
#
labels_json={}
feature_map={}#{image_name:feature}
#制作标签
labels=[]
images=[]
texts=[]
sample_neg=2
neg_num=0
pos_num=0

with open(data_dir, 'r',encoding='utf-8') as f:
    for line in tqdm(f):
        item = json.loads(line)
        #print(item)
        sample_encode=[1]#图文匹配
        keys=item['match'].keys()
        for name in class_name[1:]:
            encode=[0]
            if name in keys:#该属性匹配
                encode=[1]
            sample_encode+=encode

        labels.append(sample_encode)
        images.append(item['img_name'])
        texts.append(item['title'])
        feature_map[item['img_name']]=item['feature']
        pos_num+=1

        #-----------制作负样本----------
        '''
        通过替换标题中的词语来实现
        '''
        for _ in range(sample_neg):
            class_dict1 = {'图文': ['符合', '不符合'],
                          '版型': ['修身型', '宽松型', '标准型'],
                          '裤型': ['微喇裤', '小脚裤', '哈伦裤', '直筒裤', '阔腿裤', '铅笔裤', 'O型裤', '灯笼裤', '锥形裤', '喇叭裤', '工装裤', '背带裤',
                                 '紧身裤'],
                          '袖长': ['长袖', '短袖', '七分袖', '五分袖', '无袖', '九分袖'],
                          '裙长': ['中长裙', '短裙', '超短裙', '中裙', '长裙'],
                          '领型': ['半高领', '高领', '翻领', 'POLO领', '立领', '连帽', '娃娃领', 'V领', '圆领', '西装领', '荷叶领', '围巾领', '棒球领',
                                 '方领', '可脱卸帽', '衬衫领', 'U型领', '堆堆领', '一字领', '亨利领', '斜领', '双层领'],
                          '裤门襟': ['系带', '松紧', '拉链'],
                          '鞋帮高度': ['低帮', '高帮', '中帮'],
                          '穿着方式': ['套头', '开衫'],
                          '衣长': ['常规款', '中长款', '长款', '短款', '超短款', '超长款'],
                          '闭合方式': ['系带', '套脚', '一脚蹬', '松紧带', '魔术贴', '搭扣', '套筒', '拉链'],
                          '裤长': ['九分裤', '长裤', '五分裤', '七分裤', '短裤'],
                          '类别': ['单肩包', '斜挎包', '双肩包', '手提包']
                          }
            title=item['title']
            title_same=[]
            sample_encode=[0]#图文不匹配
            flag=0
            for name in class_name[1:]:
                encode=[0]#初始化为不匹配
                for key in item['key_attr'].keys():
                    ##出现了这个属性 决定是否将这个属性替换
                    if key==name:
                        val=item['key_attr'][key]#该属性的具体取值
                        #encode=[1]
                        if val in title:
                            #属性值在texts中，用另外的值替换掉text中文本,
                            if random.random() < 0.75:  # 制作负样本并不需要把所有属性都替换掉，只替换其中一些即可
                                tmp = class_dict[key]
                                tmp_1 = class_dict1[key]
                                # tmp_1=[]
                                dict1 = class_dict1[key]
                                val_same = []
                                # idx=class_index[key]
                                # idx=0
                                for id in range(len(class_dict[key])):
                                    if class_dict[key][id] == val:
                                        idx = class_index[key][id]

                                for id1 in range(len(class_index[key])):
                                    if idx == class_index[key][id1]:
                                        val_same.append(class_dict[key][id1])

                                for id2 in val_same:
                                    tmp_1.remove(id2)

                                sample = random.choice(tmp_1)

                                title = title.replace(val, sample)
                                encode = [0]
                                flag = 1
                                for t in range(len(title_same)):
                                    if title == title_same[t]:
                                        flag = 0
                                if flag:
                                    title_same.append(title)
                            else:  # 这个属性不被替换
                                encode = [1]
                sample_encode+=encode
            if flag==1:
                labels.append(sample_encode)
                images.append(item['img_name'])
                texts.append(title)
                neg_num+=1

#################################  制作正样本  ########################################################
            flag=1
              # 图文匹配
            title1=item['title']
            # print(sample_encode1)
            title = item['title']
            key_change=[]
            key_attr = item['key_attr']
            for name in class_name[1:]:
                   for key in key_attr.keys():
                       ##出现了这个属性 决定是否将这个属性替换
                       if key==name:
                           val=key_attr[key]#该属性的具体取值

                           if val in title:
                               key_change.append(val)
                               # if random.random() < 0.75:
            if len(key_change)>=2:
                for i in range(0,len(key_change)-1):
                    title1=title1.replace(key_change[i],key_change[i+1],1)
                    title1 = title1.replace(key_change[i+1], key_change[i], 1)
                    # flag=1
            # if len(key_change)==1:
            #     title1=title1.replace(key_change[0],'')
            #     title1=''.join((title1,key_change[0]))

                # flag=1

            if title1 == title:
                flag = 0

            if flag and num < 6000:
                # print(title1)
                num += 1
                labels.append(sample_encode)
                images.append(item['img_name'])
                texts.append(title1)
                feature_map[item['img_name']] = item['feature']
                pos_num += 1

def match_attrval(title, attr, attr_dict):
    # 在title中匹配属性值
    attrvals = "|".join(attr_dict[attr])
    ret = re.findall(attrvals, title)
    return "{}{}".format(attr, ''.join(ret))

with open(data_dir_coarse, 'r',encoding='utf-8') as f:
    for line in tqdm(f):
        item = json.loads(line)
        # print(item)
        key_attr = {}
        match = item['match']
        ku = re.findall('裤', item['title'])
        xie = re.findall('鞋', item['title'])
        xue = re.findall('靴', item['title'])
        for name in class_name[1:]:
            if name == '裤门襟':
                if ku:
                    attrvals = "|".join(class_dict[name])
                    ret = re.findall(attrvals, item['title'])
                    if ret:
                        match[name] = 1
                        key_attr[name] = ret[0]
            elif name == '闭合方式':
                if xie or xue:
                    attrvals = "|".join(class_dict[name])
                    ret = re.findall(attrvals, item['title'])
                    if ret:
                        match[name] = 1
                        key_attr[name] = ret[0]
            else:
                attrvals = "|".join(class_dict[name])
                ret = re.findall(attrvals, item['title'])
                if ret:
                    match[name] = 1
                    key_attr[name] = ret[0]
        pic = item['match']['图文']
        if pic:
            sample_encode = [1]  # 图文匹配
            pos_num += 1
        else:
            sample_encode = [0]
            neg_num += 1
        keys = match.keys()
        for name in class_name[1:]:
            encode = [0]
            if name in keys:  # 该属性匹配
                encode = [1]
            sample_encode += encode
        #
        labels.append(sample_encode)
        images.append(item['img_name'])
        texts.append(item['title'])
        feature_map[item['img_name']] = item['feature']

     #-----------制作负样本----------
        '''
        通过替换标题中的词语来实现
        '''
        if pic:
            for _ in range(sample_neg):
                class_dict1 = {'图文': ['符合', '不符合'],
                              '版型': ['修身型', '宽松型', '标准型'],
                              '裤型': ['微喇裤', '小脚裤', '哈伦裤', '直筒裤', '阔腿裤', '铅笔裤', 'O型裤', '灯笼裤', '锥形裤', '喇叭裤', '工装裤', '背带裤',
                                     '紧身裤'],
                              '袖长': ['长袖', '短袖', '七分袖', '五分袖', '无袖', '九分袖'],
                              '裙长': ['中长裙', '短裙', '超短裙', '中裙', '长裙'],
                              '领型': ['半高领', '高领', '翻领', 'POLO领', '立领', '连帽', '娃娃领', 'V领', '圆领', '西装领', '荷叶领', '围巾领', '棒球领',
                                     '方领', '可脱卸帽', '衬衫领', 'U型领', '堆堆领', '一字领', '亨利领', '斜领', '双层领'],
                              '裤门襟': ['系带', '松紧', '拉链'],
                              '鞋帮高度': ['低帮', '高帮', '中帮'],
                              '穿着方式': ['套头', '开衫'],
                              '衣长': ['常规款', '中长款', '长款', '短款', '超短款', '超长款'],
                              '闭合方式': ['系带', '套脚', '一脚蹬', '松紧带', '魔术贴', '搭扣', '套筒', '拉链'],
                              '裤长': ['九分裤', '长裤', '五分裤', '七分裤', '短裤'],
                              '类别': ['单肩包', '斜挎包', '双肩包', '手提包']
                              }
                title=item['title']
                title_same=[]
                sample_encode=[0]#图文不匹配
                flag=0
                for name in class_name[1:]:
                    encode=[0]#初始化为不匹配
                    for key in key_attr.keys():
                        ##出现了这个属性 决定是否将这个属性替换
                        if key==name:
                            val=key_attr[key]#该属性的具体取值
                            #encode=[1]
                            if val in title:
                                #属性值在texts中，用另外的值替换掉text中文本,
                                if random.random() < 0.75:#制作负样本并不需要把所有属性都替换掉，只替换其中一些即可
                                    tmp=class_dict[key]
                                    tmp_1=class_dict1[key]
                                    # tmp_1=[]
                                    dict1=class_dict1[key]
                                    val_same=[]
                                    # idx=class_index[key]
                                    # idx=0
                                    for id in range(len(class_dict[key])):
                                        if class_dict[key][id]==val:
                                            idx=class_index[key][id]

                                    for id1 in range(len(class_index[key])):
                                        if idx==class_index[key][id1]:
                                            val_same.append(class_dict[key][id1])

                                    for id2 in val_same:
                                          tmp_1.remove(id2)

                                    sample=random.choice(tmp_1)
                                    # print(sample)
                                    #print(val,sample)
                                    title=title.replace(val,sample)
                                    encode = [0]
                                    flag = 1
                                    for t in range(len(title_same)):
                                        if title==title_same[t]:
                                            flag=0
                                    if flag:
                                        title_same.append(title)
                                else:#这个属性不被替换
                                    encode=[1]
                    sample_encode+=encode
                if flag==1:
                    labels.append(sample_encode)
                    images.append(item['img_name'])
                    texts.append(title)
                    neg_num+=1

print('neg_num',neg_num)
print('pos_num',pos_num)
labels_json['label']=labels
labels_json['title']=texts
labels_json['img_name']=images
#
with open("data/label_v2.json","w") as f:
   json.dump(labels_json,f)
with open("data/feature_imgName_v2.json","w") as f:
    json.dump(feature_map,f)