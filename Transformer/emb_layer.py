import torch

# 预定义的网络层torch.nn, 工具开发者已经帮文明开发好的一些常用层
# 例如，卷积层、lstm层、embedding层等，不需要在造轮子
import torch.nn as nn

# 数学计算工具包
import math

# torch中变量封装函数
from torch.autograd import Variable

# 定义Embeddings类来实现文本前乳，这里，s说明代表两个一模一样的嵌入层，他们共享参数。
# 该类继承nn.Module，这样就有标准层的一些功能，这里我们也可以理解为一种模式。
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """类的初始化函数，两个参数，d_model：词嵌入的维度；vocab：词表大小"""
        # 然后以super的方式指明，继承nn.Module的初始化函数，我们自己实现的所有层都会这样去做
        super(Embeddings, self).__init__()
        # 之后，调用nn中预定义层Embedding,获得一个词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        # 最后，将d_model传入类中
        self.d_model = d_model

    def forward(self, x):
        """可以理解为该层的前向传播逻辑，所有层中都会有这个函数
        当传给该类的实例化对象参数时，自动调用该类函数
        参数x：因为Embedding层是首层，所以代表输入给模型的文本通过词汇映射后的张量"""

        # 将x传给self.lut并与根号下self.d_model相乘做为结果返回
        return self.lut(x) * math.sqrt(self.d_model)


if __name__ == '__main__':
    # nn.Embedding(vocab_size 词表的大小，embedding_size 输出词向量的维度，padding_idx 初始化为0）
    # nn.Embedding不会自带padding
    # embedding = nn.Embedding(10, 512)
    embedding = Embeddings(10, 512)
    print(type(embedding), embedding)
    input = torch.LongTensor([[1,2,4,5], [4,3,2,9]])
    print(type(input), input.shape, input)
    output = embedding(input)
    print(type(output), output.shape, output)

