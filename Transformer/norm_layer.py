# 它是所有深层网络模型都需要的标准网络层。因为随着网络层熟的增加，通过多层的计算后，参数可能开始出现过大或者过小。


import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        :param features: 表示词嵌入的维度
        :param eps: 一个足够小的数，在规范化公式的坟墓中出现，防止分母为0，默认是1e-6
        """

        super(LayerNorm, self).__init__()

        # 要使用nn.Parameter对其进行封装，代表它们是模型的参数，未来也会随着模型一起更新（和只定义torch.ones()不一样)
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        """x是前馈全连接层的输出"""
        # 对输入的最后一个维度求均值，并且保持与输入维度一致
        mean = x.mean(-1, keepdim=True)
        # 对输入的最后一个维度求标准差
        std = x.std(-1, keepdim=True)
        # 根据规范化公式，用x减去均值处以标准差获得规范化的
        # 最后对结果乘以我们的缩放参数，即a2，*号代表element-wise product
        return self.a2 * (x - mean) / (std + self.eps) + self.b2


if __name__ == '__main__':
    features = 512
    x = torch.randn(2,4,512)
    print(x)

    learn_norm = LayerNorm(features)
    ln_result = learn_norm(x)
    print(ln_result)