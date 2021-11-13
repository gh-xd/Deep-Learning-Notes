from emb_layer import Embeddings
from pos_enc_layer import PositionalEncoding
import torch


d_model = 512 # vec_size
dropout = 0.1
max_len = 60
vocab_size = 100

input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
print('------ Input ------\n', input)
embedding = Embeddings(d_model, vocab_size)
print('------ Embeddings ------\n', embedding)

output = embedding(input)
print('------ Output ------\n', output,'\nOutput size', output.shape)


# 实例化位置编码类
pe = PositionalEncoding(d_model, dropout, max_len)

# 将embedding层的输出，扔到实例里面，得到位置编码器的结果
pe_result = pe(output)

print('------ PE Result ------\n', pe_result, '\nPE size', pe_result.shape)