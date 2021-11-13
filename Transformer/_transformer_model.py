# all codes from integration.py

import numpy as np
import math
import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def subsequent_mask(size):
    """生成向后遮掩的掩码张量，参数size是掩码张量最后两个维度的大小，它的最后两维形成一个方阵"""
    # 在函数中，首先定义掩码张量的形状
    attn_shape = (1, size, size)

    # 然后使用np.ones方法，向这个形状中添加1元素。形成上三角阵，为了节约空间。
    # 再使其中的数据类型变为无符号8位整型unit8
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 最后，将numpy转化为torch中的tensor，内部做一个 1- 的操作
    # 目的是做一个三角阵的反转。subsequent_mask内部的每个元素都会被1减
    # 如果是0，subsequent_mask中的该位置从0变为1
    # 如果是1，subsequent_mask中的该位置由1变为0
    return torch.from_numpy(1 - subsequent_mask)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) # 视频版本，貌似这个才能起作用？！
    p_attn = F.softmax(scores, dim = -1) # 以最后一个维度进行Softmax（也就是最内层的行），size=(batch, h, L, L)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn # 与V相乘。第一个输出的size=(batch, h, L, d_model//h), 第二个输出的size=(batch,h,L,L)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0 # 检测word embedding维度是否能被h整除
        self.d_k = d_model // h # 将word_embedding分到每个头，每个头获得多少维。例如，768维的词向量，6个头，则每个head分的128维
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # 4个线性变换，前3个为QKV三个变换矩阵，最后一个用于attention之后
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # 编码器mask的size=[batch,1,1,src_L], 解码器mask的size=[batch, 1, tgt_L, tgt_L]
        nbatches = query.size(0) # 获取batch的值，nbatches = batch

        query, key, value = \
            [model(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
             for model, x in zip(self.linears, (query, key, value))] # view中给-1可以推测这个位置的维度

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1,2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x) # size = (batch, L, d_model)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()

        self.norm = LayerNorm(size)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


class EncoderDecorder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """
        :param encoder: 编码器对象
        :param decorder: 解码器对象
        :param source_embd: 源数据嵌入函数
        :param target_embd: 目标数据嵌入函数
        :param generator: 输出部分的类别生成器对象
        """
        super(EncoderDecorder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """
        :param source: 源数据
        :param target: 目标数据
        :param source_mask: 源数据的掩码张量
        :param target_mask: 目标数据的掩码张量
        :return:
        """

        # 在函数中，将source，source_mask传入编码函数，得到结果后，与source_mask，target和target_mask一起传给解码函数
        return self.decode(self.encode(source, source_mask), source_mask, target, target_mask)

    def encode(self, source, source_mask):
        """ 编码函数，以source和source_mask为参数 """
        # 使用src_embed对source进行处理，然后和source_mask一起传给self.encoder
        print('encode function type ', type(source))
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        """ 解码函数，以memory即编码器的输出，source_mask, target, target_mask为参数"""
        # 使用tgt_embed对target做处理，然后和source_mask等一起传
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


def make_model(source_vocab, target_vocab, N=6, d_model=512, d_ff=2048, head=8, dropout=0.1):
    """
    :param source_vocab: 源数据特征（词汇）总数
    :param target_vocab: 目标数据特征（词汇）总数
    :param N: 编码器和解码器的叠数
    :param d_model: 词向量印社维度
    :param d_ff: 前馈全连接网络中变换矩阵的维度
    :param head: 多注意力结构的头数
    :param dropout: 置零比率
    """

    # 首先得到一个深度拷贝命令，接下来很多结构都需要进行深度拷贝，来保证它们之间的独立性
    c = copy.deepcopy

    # 实例化多头注意力类，得到对象attn
    attn = MultiHeadedAttention(head, d_model)

    # 实例化前馈全连接层，得到对象ff
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 实例化位置编码，得到对象position
    position = PositionalEncoding(d_model, dropout)

    # 根据结构图，最外层是EncoderDecoder
    # EncoderDecoder输入：
    # 1)编码器层
    # 2)解码器层
    # 3)源数据Embedding层和位置编码层组成的有序结构，
    # 4)目标数据Embedding层和位置编码组成的有序结构
    # 5)类别生成器层
    # 在编码器层中有attention子层，以及前馈全连接层
    # 在解码器层中有两个attention子层以及前馈全连接层

    model = EncoderDecorder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一旦判断参数的维度大于1，则会将其初始化乘一个服从均匀分布的矩阵
    # for p in model.parameters()可以用于"显示模型参数更新"
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


if __name__ == '__main__':
    """ nn.init.xavier_uniform演示 """
    # 结果服从均匀分布U(-a, a)
    w = torch.empty(3,5)
    w = nn.init.xavier_uniform(w)
    print(w)


    """打印模型"""
    source_vocab = 11
    target_vocab = 11
    N = 6
    # 其他的都使用默认参数
    res = make_model(source_vocab, target_vocab, N)
    print(res)
