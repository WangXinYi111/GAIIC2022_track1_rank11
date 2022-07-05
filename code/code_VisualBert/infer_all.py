from visual_bert_model_all_var import Model      # test using visualbert
import torch
from tqdm import tqdm
import os
import json


from transformers import BertTokenizer, VisualBertModel
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# first use support_fun to change all titles in text
model_save_dir_pth = "/home1/wxwHD/GAIIC_track1_baseline/Baseline_variant/change_embedding_change_cls_visual_bert.pth"
weight_path = "/home1/wxwHD/GAIIC_track1_baseline/Baseline_variant/Visual_model_temp/fold_1_best.pth"


def load_model(weight_path, visualbert_model):
    print(weight_path)
     ##
    model = Model(visualbert_model, tokenizer)
    model.load_state_dict(torch.load(weight_path))
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def predict(text, feature, attention):
    outputs = model(text, attention, feature)
    outputs = outputs.sigmoid().detach().cpu().numpy()[0]      # use sigmoid to map output to (0, 1)
    return outputs


if __name__=="__main__":
    #
    out_file = 'result_folder/temp.txt'

    device = torch.device('cuda')
    class_name = ['图文','版型', '裤型', '袖长', '裙长', '领型', '裤门襟', '鞋帮高度', '穿着方式', '衣长', '闭合方式', '裤长', '类别']
    class_dict = {'图文': ['符合', '不符合'],
                  '版型': ['宽松型', '标准型'],
                  '裤型': ['直筒裤', '阔腿裤', '锥形裤', '喇叭裤', '工装裤', '背带裤', '紧身裤'],
                  '袖长': ['七分袖', '五分袖', '零分袖', '九分袖'],
                  '裙长': ['短裙', '中裙', '长裙'],
                  '领型': ['高领', '连帽', 'V领', '圆领', '西装领', '围巾领', '棒球领', '方领', 'U型领', '堆堆领', '一字领', '亨利领', '斜领', '双层领'],
                  '裤门襟': ['系带', '松紧', '拉链'],
                  '鞋帮高度': ['低帮', '高帮'],
                  '穿着方式': ['套头', '开衫'],
                  '衣长': ['中长款', '超短款', '超长款'],
                  '闭合方式': ['系带', '松紧带', '魔术贴', '搭扣', '套筒', '拉链'],
                  '裤长': ['九分裤', '五分裤', '七分裤', '零分裤'],
                  '类别': ['单肩包', '斜挎包', '双肩包', '手提包']
                  }
    submit_sample = {"img_name":"test000255","match":{"图文":0,"领型":1,"袖长":1,"穿着方式":0}}
    submit = []
    class_index = []
    model_list = []
    num = 0
    num1 = 0
    num2 = 0
    num3 = 0
    #
    tokenizer = BertTokenizer.from_pretrained("/home1/wxwHD/GAIIC_track1_baseline/Bert_wwm/chinese_wwm_pytorch")
    visualBert_model = torch.load(model_save_dir_pth)
    model = load_model(weight_path, visualBert_model)

    Threshold = 0.50    #
    data_dir = "/home1/wxwHD/GAIIC_track1_baseline/Baseline_variant/test_B.txt"
    # "/home1/wxwHD/GAIIC_track1_baseline/New_baseline/data/test.txt"
    with open(data_dir, 'r') as f:
        for line in tqdm(f):

            item = json.loads(line)
            img_name = item['img_name']
            title = item['title']
            query = item['query']
            feature = torch.tensor(item['feature'])
            feature = feature.unsqueeze(0)      # 1 * 2048
            feature = feature.cuda().float()
            outputs = model(title, feature)

            tmp = {}
            for que in query:
                inx = class_name.index(que)
                if outputs.logits[0, inx] > 0:
                    tmp[que] = 1
                    # if outputs.logits[0, inx] < 1:
                    #     num2 += 1
                    # print(outputs.logits[0, inx])
                else:
                    # if outputs.logits[0, inx] > -1:
                    #     # print(outputs.logits[0, inx])
                    #     num += 1
                    # elif outputs.logits[0, inx] > -2:
                    #     num1 += 1
                    # elif outputs.logits[0, inx] > -3:
                    #     num3 += 1

                    tmp[que] = 0

            submit_sample['img_name'] = img_name
            submit_sample['match'] = tmp

            submit.append(json.dumps(submit_sample, ensure_ascii = False)+'\n')
    #
    with open(out_file, 'w') as f:
        f.writelines(submit)
    # print("-1-0 value" + str(num))
    # print("-2--1 value" + str(num1))
    # print("-3--2 value" + str(num3))
    # print("1-2 value" + str(num2))