import torch
import matplotlib.pyplot as plt
# 预定义的网络层torch.nn, 工具开发者已经帮文明开发好的一些常用层
# 例如，卷积层、lstm层、embedding层等，不需要在造轮子
import torch.nn as nn

# 数学计算工具包
import math

# torch中变量封装函数
from torch.autograd import Variable

# 定义位置编码器类，同样也是一个层，所以继承nn.Module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        """位置编码器类初始化函数，有三个参数，d_model 词嵌入维度，dropout 置0比率，max_len 每个句子的最大长度"""
        super(PositionalEncoding, self).__init__()

        # 实例化nn中预定义的Dropout层，并将dropout传入其中，获得对象self.dropout
        self.dropout = nn.Dropout(p=dropout)

        # 初始一个位置编码矩阵，它是一个0矩阵，矩阵的大小为Max_len x d_model
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵，这里，词汇的绝对位置就是它的索引
        # 所以，首先使用arange方法获得一个连续自然数向量，然后使用unsequeeze方法拓展维度
        # 因为参数传的是1，代表矩阵拓展的位置，会使向量变成一个max_len x 1 的矩阵
        position = torch.arange(0, max_len).unsqueeze(1)

        # 接下来，考虑如何将位置信息加入到位置编码
        # 简单思路：将max_len x 1的绝对位置矩阵，变换成max_len x d_model形状，覆盖
        # 要这么做，九需要一个 1 x d_model 形状的变换矩阵div_term
        # 还希望它能将自然数的绝对编码缩放成了足够小的数字，有助于梯度下降过程
        # 首先，使用一个arrange获得一个自然数矩阵
        # 先初始化一个初始化一半的 1 x m_model /2 的矩阵，为什么呢？
        # 等于是初始化了两次，每次初始化的变换矩阵会做不同的处理
        # 两个矩阵分别填充在位置编码矩阵的偶数和奇数位置上，组成最终的位置编码矩阵
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 这样我们就得到了位置编码矩阵pe。pe目前还只是二维的，想要和embedding的输出一样，必须拓展
        # 所以，使用unsqueeze拓展维度。
        pe = pe.unsqueeze(0)

        # 把pe的位置编码矩阵注册成模型的buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        """forward函数的参数是x，表示文本序列的词嵌入表示"""
        # 对pe做一些适配，将这三个张量的第二维（句子最大长度）切片
        # 我们默认max_len为5000，一半太大，很难有一条句子5000个词
        # 最后使用Variable进行封装，与x的央视相同。但不需要进行梯度求解
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        # 最后使用self.dropout对象进行丢弃操作
        return self.dropout(x)


if __name__ == '__main__':
    # abc = torch.arange(0, 100)
    # abc_ = abc.unsqueeze(1)
    # print('torch.arange Type: ', type(abc), '|torch.arange.unsequeeze Type: ', type(abc_))
    # print('torch.arange Size: ', abc.size(), '|torch.arange.unsequeeze Size: ', abc_.size())

    # m = nn.Dropout(p=0.2)
    # input = torch.randn(4,5)
    # print("[Input] - ", input, "[Input Type]", type(input), input.size())
    # output = m(input)
    # print("[Output] - ", output, "[Output Type]", type(output), input.size())

    a = PositionalEncoding(256, 0.2)
    ape = a.pe
    s_ape = ape.squeeze(0)
    s_ape_size = tuple(s_ape.shape)
    print(s_ape)
    print(s_ape_size)

    # plt.plot(s_ape, 'o')
    # plt.show()

    # fig = plt.figure(figsize=s_ape_size)
    # ax = plt.gca()
    # cas = plt.imshow(s_ape, cmap='viridis')
    # plt.show()