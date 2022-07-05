import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from math import sqrt
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=4):
        super(MultiHeadSelfAttention, self).__init__()
        # 维度必须能被num_head 整除
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        # 定义线性变换矩阵
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att


class CNN_Text(nn.Module):

    def __init__(self, embed_num, class_num=13, dropout=0.25):
        super(CNN_Text, self).__init__()
        self.attention = MultiHeadSelfAttention(256, 256, 256)
        embed_dim = 256
        Ci = 1
        kernel_num = 100
        Ks = [2, 3, 4, 5]

        self.embed = nn.Embedding(embed_num, embed_dim)  # 词嵌入
        self.convs = nn.ModuleList([nn.Conv2d(Ci, kernel_num, (K, embed_dim)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        # self.fusion_a = nn.Linear(256,400)
        # self.fusion_b = nn.Linear(256,400)
        #
        self.img_head = nn.Sequential(
            nn.Linear(2048, 256),
            # nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.LayerNorm(256)

        )
        self.classify = nn.Sequential(


            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, class_num)
        )
        #

    def forward(self, x, frt):
        x = self.embed(x)  # (N, W, D)-batch,单词数量，维度
        x_img = self.img_head(frt)
        x_img = x_img.unsqueeze(1)
        x = torch.cat([x, x_img], axis=1)  # 融合V，L模块
        ateion = self.attention(x)
        # ateion = ateion.squeeze(1)
        ateion = ateion.transpose(1, 2)
        y = F.avg_pool1d(ateion, ateion.size(2)).squeeze(2)
        x = F.max_pool1d(ateion, ateion.size(2)).squeeze(2)  # [(N, Co), ...]*len(Ks)
        x = x + y
        # x = x.unsqueeze(0)
        logit = self.classify(x)  # (N, C)
        return logit
if __name__=="__main__":
    net=CNN_Text(embed_num=1000)
    x=torch.LongTensor([[1,2,4,5,2,35,43,113,111,451,455,22,45,55],[14,3,12,9,13,4,51,45,53,17,57,954,156,23]])
    frt=torch.randn(2,2048)
    logit=net(x,frt)
    print(logit.shape)
