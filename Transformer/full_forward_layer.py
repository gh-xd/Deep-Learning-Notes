# 考虑到注意力机制可能对复杂过程的拟合程度不够，通过增加两层网络来增强模型的能力
# 这是一个实验结果。。。

import torch
import torch.nn as nn
import torch.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """初始化函数有三个输入参数分别是d_mode, d_ff和dropout=0.1
        第一个是线性层的输入维度，因为我们希望输入通过前馈全连接层后，输入和输出唯独不变
        第二个参数d_ff是第二个线性变换"""

        super(PositionwiseFeedForward, self).__init__()

        # 首先按照我们预期使用nn实例化了两个线性层对象，self.w1和self.w2
        # 它们的参数分别是d_model, d_ff和d_ff, d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

        # 然后使用nn的Dropout实例化对象self.dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """输入参数为x，代表来自上一层的输出，多头注意力的输出"""
        # 首先经过第一个线性层。然后使用Functional中relu函数激活
        # 之后，使用dropout进行随机置0，最后通过第二个线性层w2，返回结果
        return self.w2(self.dropout(F.relu(self.w1(x))))


if __name__ == '__main__':
    a = torch.randn(2,3,5)
    print(a)
    a = a.view(2, -1)
    print(a)
    print(a.shape)
    Ln = nn.Linear(15,8)
    b = Ln(a)
    print(b)
    print(b.shape)