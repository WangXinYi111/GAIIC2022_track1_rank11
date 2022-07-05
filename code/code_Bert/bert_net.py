import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertTokenizer
from transformers import BertPreTrainedModel, BertModel
import os
# comparison

class Model(nn.Module):

    def __init__(self, bert_model, class_num = 14, dropout = 0.25):
        super(Model, self).__init__()
        #
        Ks = [3, 4, 5]
        in_channel = 1
        out_channel = 100
        embedding_dim = 768
        self.convs1 = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, embedding_dim)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        # self.mapping = nn.Linear(len(Ks) * out_channel, opt.final_dims)
        #

        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        #
        self.img_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(256),

        )
        self.classify = nn.Sequential(
            #
            # nn.Linear(556, 512),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(),
            # nn.Dropout(p=dropout),
            #
            nn.Linear(556, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(512),

            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(256),

            nn.Linear(256, class_num)
        )
        #
    def forward(self,  x,  attention, frt):   # N * 2048   # N * 768
        all_encoder_layers = self.bert(x, attention_mask = attention)

        x = all_encoder_layers["last_hidden_state"].unsqueeze(1)  # (batch_size, 1, token_num, embedding_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(batch_size, out_channel, W), ...]*len(Ks)
        y = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        d = []
        for i,j in zip(x,y):
            summ = i +j
            d.append(summ)
        x = torch.cat(d,1)

        x_img = self.img_head(frt)
        x = torch.cat([x, x_img], axis=1)  # (64, 300) + (64, 256)
        logit = self.classify(x)  # (N, C) # (64, 13)

        return logit


if __name__ == "__main__":


    tokenizer = BertTokenizer.from_pretrained("/home1/wxwHD/GAIIC_track1_baseline/Bert_wwm/chinese_wwm_pytorch")
    model = BertModel.from_pretrained("/home1/wxwHD/GAIIC_track1_baseline/PretrainingBERT-main/model")
    # model = model.load_state_dict(torch.load("/home1/wxwHD/GAIIC_track1_baseline/PretrainingBERT-main/model/pytorch_model.bin"))
    # embedded = bert_model("词语来实现")[1]

    net = Model(model)

    x = torch.LongTensor([[1, 2, 4, 5, 2, 35, 43, 113, 111, 451, 455, 22, 45, 55],
                          [14, 3, 12, 9, 13, 4, 51, 45, 53, 17, 57, 954, 156, 23]])

    attention = torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

    frt = torch.randn(2, 2048)

    logit = net(x, attention, frt)
    print(logit.shape)