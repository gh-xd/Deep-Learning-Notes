

# naive transformer问题：
# 加batch，就是n x d 变成 b x n x d
# 加dropout
# multihead
# meask padding
#    不能保证batch都是等长，需要padding为定长。

import torch
import torch.nn as nn
import math


class NaiveTransformerLayer(nn.Module):

    def __init__(self):
        self.dim = 768
        self.att_drop_prob = 0.1 # 为添加Dropout层设置的参数
        self.state_drop_prob = 0.5 # 为添加Dropout层设置的参数

        self.Wq = nn.Linear(self.dim, self.dim, bias=False)
        self.Wk = nn.Linear(self.dim, self.dim, bias=False)
        self.Wv = nn.Linear(self.dim, self.dim, bias=False)
        self.lm = nn.LayerNorm(self.dim)
        self.ffn1 = nn.Linear(self.dim, self.dim*4)
        self.ffn2 = nn.Linear(self.dim*4, self.dim)
        self.act = nn.GELU()
        self.lm_ffn = nn.LayerNorm(self.dim)

        self.att_drop = nn.Dropout(self.att_drop_prob) # 为attention层添加的Dropout
        self.state_drop = nn.Dropout(self.state_drop_prob) # 为？层添加的Dropout

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

        #
        attention_score = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.dim) # mm -> bmm，转置的维度改成(1,2)
        attention_score = nn.Softmax(dim=2)(attention_score) # dim改成2。dim指的是，在第几维上加起来等于1
        attention_score = self.att_drop(attention_score) # 添加dropout层
        O = torch.bmm(attention_score, V) # mm -> bmm
        O = self.state_drop(O) # 添加dropout层
        O = self.lm(x + O)
        return O
    
    def FFN(self, x):
        hidden = self.act(self.ffn1(x))
        output = self.ffn2(hidden)
        output = self.state_drop(output) # 添加dropout层
        output = self.lm_ffn(x + output)
        return output

    def forward(self, x):
        """
        input x:  b x n x d
        output: b x  n x d
        :return:
        """
        x = self.SelfAttention(x)
        x = self.FFN(x)

        return x


class MultiTransformerLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.dim = 768
        self.att_drop_prob = 0.1  # 为添加Dropout层设置的参数
        self.state_drop_prob = 0.5  # 为添加Dropout层设置的参数
        self.num_heads = 12 # bert默认12个head
        self.size_per_head = self.dim // self.num_heads # 每个head的维度，是dim整除head数，一般是64

        # DIFFERENCE TO BertSelfAttention: query, key, value是有偏置项的（即，没有设置bias=False)
        self.Wq = nn.Linear(self.dim, self.num_heads * self.size_per_head, bias=False) # 线性层的维度改变乘和head有关的
        self.Wk = nn.Linear(self.dim, self.num_heads * self.size_per_head, bias=False)
        self.Wv = nn.Linear(self.dim, self.num_heads * self.size_per_head, bias=False)

        self.W = nn.Linear(self.num_heads * self.size_per_head, self.dim) # 加一个linear层，保证ResNet里的维度是正确的。

        self.lm = nn.LayerNorm(self.dim)
        self.ffn1 = nn.Linear(self.dim, self.dim * 4)
        self.ffn2 = nn.Linear(self.dim * 4, self.dim)
        self.act = nn.GELU()
        self.lm_ffn = nn.LayerNorm(self.dim)

        self.att_drop = nn.Dropout(self.att_drop_prob)  # 为attention层添加的Dropout
        self.state_drop = nn.Dropout(self.state_drop_prob)  # 为？层添加的Dropout

    def calc_mask_score(self, attention_mask):
        # 假设所有的head都是公用一个mask
        """
        attention_mask input (b,n)
        output (b,h,n,n) 的score --> 对于每一个head任意两个部分之间，是否应该被注意到

        [1 1 1 1 ... 0 0]
        通过简单的矩阵广播来实现？

        :param attention_mask:
        :return:
        """
        mask_score = torch.zeros(attention_mask.size(0), self.num_heads, attention_mask.size(1), attention_mask.size(1))
        mask_score = mask_score + attention_mask[:, None, None, :]

        # DIFFERENCE TO BertSelfAttention: HF没有做以下这步
        mask_score = (1.0 - mask_score) * -10000.
        return mask_score

    def SelfAttention(self, x, attention_mask):
        """
        这是一个Self-attention层
        input n x d
        output n x d

        Q, K, V的维度需要进行改变。
            batch * n * num_head * size_per_head
            如果需要每个head进行单独的计算（但有希望并行运算所有的head），就需要进行维度的变换：
            batch * n * num_head * size_per_head --> batch * num_head * n * size_per_head (b,h,n,s)
            (n,s)是self-attention需要做的，所以把h移到前面，每个head可以单独做自己的计算，同时所有head并行

        attention_mask:
            保持和HF一样：
                1 代表考虑，normal token
                0 代表不考虑，masked token
            形状 (b,n)
            需要一个新的函数calc_mask_score计算masked score

        :param x:
        :return:
        """
        # new_size的处理：去掉(b,n,d)的d这一维，加上(h,s)维。s本身就是d整除h得出来的数。
        new_size = x.size()[:-1] + (self.num_heads, self.size_per_head) # (b,n,h,s)
        Q = self.Wq(x).view(*new_size).permute(0,2,1,3) # view是变维度，从Q的(b,n,d) --> Q的(b,n,h,s) --> (b,h,n,s)
        K = self.Wk(x).view(*new_size).permute(0,2,1,3) # 这样，每个K就被分割
        V = self.Wv(x).view(*new_size).permute(0,2,1,3) # 这样，每个V就被分割

        # DIFFERENCE TO BertSelfAttention: transpose(-1,-2)

        attention_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.dim)  # bmm不适应4维，需要换成matmul。transpose第3和第4个维度，即(n,s)

        # 这里加入mask, attention mask here。在softmax之前，将需要mask的地方，减掉一个很大的数字-->趋近于零-->不考虑
        # 在def SelfAttention(self,x)需要加一个输入attention_mask
        attention_score = attention_score + self.calc_mask_score(attention_mask)

        # DIFFERENCE TO BertSelfAttention: softmax(dim=-1)
        attention_score = nn.Softmax(dim=3)(attention_score)  # dim改成3。dim指的是，在第几维上加起来等于1
        attention_score = self.att_drop(attention_score)  # 添加dropout层
        O = torch.matmul(attention_score, V)  # bmm不适应4维，需要换成matmul

        # O的形状：(b,h,n,s)。但是，要进行ResNet的话，维度还得调回(b,n,d)，需要重新排。
        # 但是，(b,h,n,s)转化为(b,n,h,s)以后，h*s并不一定会是d，因为s=d//h，所以需要加一层linear，来保证这个事情。
        # DIFFERENCE TO BertSelfAttention: permute(...)后面加了.contiguous()。由于permute在内存里并没有创造新顺序的tensor，contiguous保证它不会出问题。
        O = self.W(O.permute(0,2,1,3))
        O = self.state_drop(O) # O.shape (b,n,d)
        O = self.lm(x + O)
        return O

    def FFN(self, x):
        hidden = self.act(self.ffn1(x))
        output = self.ffn2(hidden)
        output = self.state_drop(output)  # 添加dropout层
        output = self.lm_ffn(x + output)
        return output

    def forward(self, x):
        """
        input x:  b x n x d
        output: b x  n x d
        :return:
        """
        x = self.SelfAttention(x)
        x = self.FFN(x)

        return x