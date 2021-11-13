# nn.functional 工具包装载了网络层中那些只进行计算，而没有参数的层
import torch.nn.functional as F
import torch.nn as nn
import torch


# 将线性层和softmax计算层一起实现，因为二者的共同目标是生成最后的结构
# 因此，把类的名字叫做Generator，生成器类


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """
        :param d_model: 词嵌入维度
        :param vocab_size: 词表大小
        """
        super(Generator, self).__init__()
        # 首先使用nn中的预定义线性层进行实例化，得到一个对象self.project等待使用
        # 该线性层有两个参数
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """ 前向逻辑函数中输入的是上一层的输出张量x"""
        # 函数中，首先使用上一步得到的self.project对x进行线性变化
        # 然后使用F的log_softmax进行softmax处理，使用它主要是因为和pytorch版本的顺势函数实现有关
        # log_softmax九是对softmax的结果取对数，因为对数函数是单调递增函数，
        # 其实对我们取最大的概率值没有影响，最后返回结果
        return F.log_softmax(self.project(x), dim=-1)


if __name__ == '__main__':
    """ nn.Linear 演示 """
    # m = nn.Linear(20, 30)
    # input = torch.randn(128, 20)
    # output = m(input)
    # print(output.size())

    """ Generator类演示"""
    d_model = 512
    vocab_size = 1000

