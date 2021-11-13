import copy
import torch
import math

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from int_sub_encoder_connection import SublayerConnection

def clones(module, N):
    # 工具人函数，定义N个相同的模块
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """
    实现 Scaled Dot-Product Attention
    :param query: 输入与Q矩阵相乘后的结果，size=(batch, h, L, d_model//h)
    :param key: 输入与K矩阵相乘后的结果，size同上
    :param value: 输入与V矩阵相乘后的结果，size同上
    :param mask: 掩码矩阵
    :param dropout: dropout率
    """
    # 取query最后一个维度的大小。取词向量一个头的维度大小
    d_k = query.size(-1)

    # 按注意力公式，将query与key的转置相乘。这里面key是将最后两个维度进行转置，再除以缩放系数，得到scores
    # torch.transpose文档 => torch.transpose(input, dim0, dim1, out=None) 交换维度dim0（默认0）和dim1（默认1），只能两维。
    # 转置功能相同的另一个是 permute(dims) with dims (int*) int*为换位顺序，可以多维。
    print(query.shape, key.shape)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 是否使用mask掩码张量
    if mask is not None:
        # 掩码矩阵，编码器mask的size=[batch, 1, 1, src_L], 解码器mask的size= [batch, 1, tgt_L, tgt_L]
        # scores = scores.masked_fill(mask=mask, value=torch.tensor(-1e9))
        print(scores.shape, mask.shape)
        scores = scores.masked_fill(mask == 0, -1e9) # 视频版本，貌似这个才能起作用？！
    p_attn = F.softmax(scores, dim = -1) # 以最后一个维度进行Softmax（也就是最内层的行），size=(batch, h, L, L)
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 最后，根据公式将p_attn与value张量相乘，获得最终query注意力的表示。同时也返回注意力张量p_attn
    return torch.matmul(p_attn, value), p_attn # 与V相乘。第一个输出的size=(batch, h, L, d_model//h), 第二个输出的size=(batch,h,L,L)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        shi实现多头注意力机制
        :param h: 头数
        :param d_model: word embedding 维度
        :param dropout: dropout率
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0 # 检测word embedding维度是否能被h整除
        # We assume d_v always equals d_k
        # 得到每个头获得的分割词向量的维度d_k
        self.d_k = d_model // h # 将word_embedding分到每个头，每个头获得多少维。例如，768维的词向量，6个头，则每个head分的128维
        self.h = h
        # 通过nn的Linear实例化，内部变换矩阵是 d_model x d_model（例子：512 x 512)
        # 我们需要4个全连接层，3个给QKV，1个给concat
        # 注意！！！ Linear(x,y)里的x和y指的是weights的维度，也就是说，input为(5,3)、weight(linier)为(3, 1)的话，输出为(5,1)
        self.linears = clones(nn.Linear(d_model, d_model), 4) # 4个线性变换，前3个为QKV三个变换矩阵，最后一个用于attention之后
        # attn代表最后得到的注意力张量
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        :param query: 输入x，即（word embedding + positional embedding), size=[batch, L, d_model]，tip: 编解码器的L可能不同
        :param key: 同上，size同上
        :param value: 同上，size同上
        :param mask: 掩码矩阵，编码器mask的size=[batch, 1 ,src_L], 解码器的size = [batch, tgt_L, tgt_L]
        """
        if mask is not None:
            # 在"头"的位置增加维度，为所有头执行相同的mask操作
            mask = mask.unsqueeze(1) # 编码器mask的size=[batch,1,1,src_L], 解码器mask的size=[batch, 1, tgt_L, tgt_L]
        # 传入多少条样本
        nbatches = query.size(0) # 获取batch的值，nbatches = batch

        """核心部分"""
        # 1) 利用三个全连接算出QKV向量，再维度变换 [batch, L, d, d_model] ---> [batch, h, L, d_model//h]
        # 首先，利用zip将输入QKV与三个现行层组合到一起
        # 然后，使用for循环，将输入QKV分别传入到线性层中。
        # 之后，开始为每个头分割输入，这里使用view方法对线性变换结果进行维度重塑（view函数作用为重构张量维度，相当于numpy中的resize）
        # 打比方 tt1 = [1,2,3,4,5,6], tt1.view(3,2) --> [[1,2], [3,4], [5,6]]
        # 意味着，每个头可以获得一部分词特征组成的的句子（部分词向量表达了一个句子的部分"语义"）
        # -1 代表自适应维度（自适应后，等于句子长度）
        # 然后，对2和3维进行转置操作，为了让代表句子长度维度（-1）和词向量维度（d_k)能够相邻，从而注意力机制才能找到词义与句子位置的关系
        # 从attention函数可以看到，利用的是原始输入的倒数第一和第二维，
        query, key, value = \
            [model(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
             for model, x in zip(self.linears, (query, key, value))] # view中给-1可以推测这个位置的维度

        # 2）实现scaled dot-product attention。x的size=(batch, h, L, d_model//h), attn的size=(batch, h, L, L)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 这步实现拼接。transpose的结果 size = (batch, L, h, d_model//h), view的结果 size = (batch, L, d_model)
        # continguous方法主要是为了能够让转置后的张量应用view方法，否则将无法直接使用
        # 然后，将其应用view方法
        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # 使用线性层列表中的最后一个线性层对输入进行线性变换，得到最终的多头注意力结构的输出
        return self.linears[-1](x) # size = (batch, L, d_model)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """

        :param size: 词前乳的维度大小
        :param self_attn: 多头注意力子层实例化对象
        :param feed_forward: 前馈全连接层实例化对象
        :param dropout:
        """
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


if __name__ == '__main__':
    # input = Variable(torch.randn(5,5))
    # print(input)
    #
    # mask = Variable(torch.zeros(5,5))
    # print(mask)
    #
    # m_input = input.masked_fill(mask=mask, value=torch.tensor(-1e9))
    # print(m_input)

    a = torch.randn(5,3)
    print(a)

    linne = nn.Linear(3,1)
    print(linne(a))