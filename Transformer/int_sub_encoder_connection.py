# 使用了残差连接
# 用一个类来处理子层连接

import torch
import torch.nn as nn
from torch.autograd import Variable

from norm_layer import LayerNorm
from encoder_layer import MultiHeadedAttention

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        """
        :param size: 词嵌入维度的大小
        :param dropout:
        """
        super(SublayerConnection, self).__init__()

        # 实例化规范化对象self.norm
        self.norm = LayerNorm(size)

        # 使用nn中预定义的dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        """前向逻辑函数中，接收上一个层或者子层的输入作为第一个参数，
        将该子层连接中的子层函数作为第二个参数"""

        # 我们首先对输出进行规范化，然后将结果传给子层处理，之后再对子层进行dropout操作
        # 随机停止一些网络中神经元的作用，来防止过拟合。最后还有一个add操作，
        # 因为存在跳跃连接，所以是将输入x与dropout后的子层输出结果相加作为最终的子层连接输出
        return x + self.dropout(sublayer(self.norm(x)))

if __name__ == '__main__':
    size = 512
    dropout = 0.2
    head = 8
    d_model = 512

    x = torch.randn(2,4,512) # x = pe_result
    mask = Variable(torch.zeros(2, 4 ,4))

    # 假设子层中装的是多头注意力层，实例化这个类
    self_attn = MultiHeadedAttention(head, d_model)

    # 使用lambda获得一个函数类型的子层
    sublayer = lambda x: self_attn(x, x, x, mask=mask)

    sc = SublayerConnection(size=size, dropout=dropout)
    print(sc, type(sc))
    sc_result = sc(x, sublayer)
    print(type(sc_result))

    # print(sc_result)
    # print(sc_result.shape)
