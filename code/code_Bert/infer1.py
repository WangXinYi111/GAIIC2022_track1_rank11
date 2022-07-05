from bert_net import Model
import torch
from tqdm import tqdm
import os
import json

from transformers import BertTokenizer, BertModel
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# first use support_fun to change all titles in text

def load_model(weight_path, bert_ori):
    print(weight_path)
     ##
    model = Model(bert_ori, class_num = 13)
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
    out_file = 'data/tmp_data/Bert_result1.txt'

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

    #
    tokenizer = BertTokenizer.from_pretrained("data/pretrain_model/chinese_wwm_pytorch")
    bert_ori = BertModel.from_pretrained("data/pretrain_model/2nd_pretrain_Bertmodel")
    model = load_model('data/best_model_Bert1.pth', bert_ori)
    # model = load_model('ckpt_v12_new_model/fold_' + str(1) + '_best.pth', bert_ori)

    Threshold = 0.50    #
    data_dir = "data/contest_data/preliminary_testB.txt"
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

            encoding = tokenizer.encode_plus(
                title,
                max_length = 40,
                add_special_tokens = True,  # Add '[CLS]' and '[SEP]'
                return_token_type_ids = False,
                padding = "max_length",
                return_attention_mask = True,
                return_tensors = 'pt',  # Return PyTorch tensors
            )

            text = torch.tensor(encoding["input_ids"][0])
            text = text.unsqueeze(0)     # 1 * 768
            text = text.type(torch.LongTensor).cuda()
            attention = torch.tensor(encoding["attention_mask"])
            attention = attention.type(torch.LongTensor).cuda()

            pre = predict(text, feature, attention)
            tmp = {}
            for que in query:
                inx = class_name.index(que)
                if pre[inx] > Threshold:
                    tmp[que] = 1
                else:
                    tmp[que] = 0
            submit_sample['img_name'] = img_name
            submit_sample['match'] = tmp

            # print(submit_sample)
            submit.append(json.dumps(submit_sample, ensure_ascii = False)+'\n')
    #
    with open(out_file, 'w') as f:
        f.writelines(submit)
