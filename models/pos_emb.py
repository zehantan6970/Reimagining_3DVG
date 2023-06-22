import torch
import math
device = "cuda" if torch.cuda.is_available() else "cpu"

class AbsolutePositionalEncoding(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq_len):
        super(AbsolutePositionalEncoding, self).__init__()

        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # 为每个位置编码生成一组正弦和余弦函数
        pos_mat=torch.arange(max_seq_len).reshape((-1,1))
        i_mat=torch.pow(10000,torch.arange(0,embedding_dim,2).reshape((1,-1))/embedding_dim)
        pe_embedding_table = torch.zeros(max_seq_len, embedding_dim)
        pe_embedding_table[:,0::2]=torch.sin(pos_mat/i_mat)
        pe_embedding_table[:,1::2]=torch.cos(pos_mat/i_mat)
        pe_embedding_table=torch.unsqueeze(pe_embedding_table,0)

        # 为每个位置编码创建一个可学习的参数
        self.register_buffer('pe_embedding_table', pe_embedding_table)

    def forward(self, batch_size, seq_len ):

        if seq_len > self.max_seq_len:
            raise ValueError("Sequence length exceeds maximum sequence length")

        # 从可学习的参数中提取每个位置的位置编码
        positional_encoding = self.pe_embedding_table[:seq_len, :]
        positional_encoding = positional_encoding.expand(batch_size, -1, -1).to(device)

        # 将位置编码添加到输入张量中
        return positional_encoding
