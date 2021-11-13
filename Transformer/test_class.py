import torch


## __call__的功能
# class Jump():
#     def __init__(self, a, b):
#         self.a = a
#         self.b = b
#
#     def __call__(self, *args, **kwargs):
#         if type(*args) is int:
#             return self.sum(*args)
#
#
#     def sum(self, c):
#         return self.a + self.b + c
#
#
#     # def multiple(self):
#     #     print(self.a * self.b)
#
# if __name__ == '__main__':
#     x = Jump(2,3)
#     print(x)
#     b = x(3)
#     print(b)


# torch.size | torch.size(0) 代表tensor第0维的数字，其他同理。

a = [[1,2],[2,3],[3,4],[4,5]]
at = torch.tensor(a)
print(at.size())