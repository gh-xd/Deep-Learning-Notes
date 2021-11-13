# naive transformer
"""
缺点： 没有batch，这个模型不能处理batch；没有dropout；没有multi-head attention；也没有attention mask / padding mask


"""


import torch
import torch.nn as nn
import math


class NaiveTransformerLayer(nn.Module):

    def __init__(self):
        self.dim = 768
        self.Wq = nn.Linear(self.dim, self.dim, bias=False)
        self.Wk = nn.Linear(self.dim, self.dim, bias=False)
        self.Wv = nn.Linear(self.dim, self.dim, bias=False)
        self.lm = nn.LayerNorm(self.dim)
        self.ffn1 = nn.Linear(self.dim, self.dim*4)
        self.ffn2 = nn.Linear(self.dim*4, self.dim)
        self.act = nn.GELU()
        self.lm_ffn = nn.LayerNorm(self.dim)

    def SelfAttention(self, x):
        """
        这是一个Self-attention层
        input n x d
        output n x d

        :param x:
        :return:
        """
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        attention_score = torch.mm(Q, K.transpose(0, 1)) / math.sqrt(self.dim)
        attention_score = nn.Softmax(dim=1)(attention_score)
        O = torch.mm(attention_score, V)
        O = self.lm(x + O)
        return O
    
    def FFN(self, x): # Feed forward network - 双层mlp，两个linear层，扩大四倍，再变回来。需要有bias（默认True）
        hidden = self.act(self.ffn1(x))
        output = self.ffn2(hidden)
        output = self.lm_ffn(x + output)
        return output

    def forward(self):
        """
        input x: n x d
        output n x d
        :return:
        """
        x = self.SelfAttention(x)
        x = self.FFN(x)

        return x