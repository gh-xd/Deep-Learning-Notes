import torch.nn as nn
from int_sub_encoder_connection import SublayerConnection
from encoder_layer import MultiHeadedAttention
from full_forward_layer import PositionwiseFeedForward
import torch

def clones(module, N):
    # 工具人函数，定义N个相同的模块
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        :param size: 词嵌入的维度，解码器的尺寸
        :param self_attn: 多头自注意力对象，也就是说这个注意力机制需要Q=K=V
        :param src_attn: 多头注意力对象，这里Q!=K=V
        :param feed_forward: 前馈全连接层对象
        :param dropout:
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 按照结构图使用clones函数克隆三个子层连接对象
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """
        :param x: 上一层输入x
        :param memory: 来自编码器的语义储存变量memory
        :param source_mask: 源数据掩码张量
        :param target_mask: 目标数据掩码张量
        """

        m = memory

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # 第二个子层，常规注意力机制，q是输入x，k和v是编码层输出memory
        # 同样也传入source_mask, 但是进行源数据遮掩的原因并非是抑制信息泄漏，而是遮掩掉对结果没有意义的值。
        # 从而提升模型效果和训练速度

        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        # 最后一个子层是前馈全连接子层
        return self.sublayer[2](x, self.feed_forward)

if __name__ == '__main__':
    head = 8
    size = 512
    d_model = 512
    d_ff =64
    dropout = 0.2
    self_attn = src_attn = MultiHeadedAttention(head, d_model, dropout)

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)


    x = torch.randn(2,4)

    # memory是来自编码器的输出
    memory = en_result