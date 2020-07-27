# coding=utf-8
import torch as t
import torch.nn as nn
from .BasicModule import BasicModule
from torchcrf import CRF
import os


# 项目根目录
ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '../'))


class BiLSTM_CRF(BasicModule):
    """
    encoder: 双向LSTM
    decoder: CRF
    """
    def __init__(self, vocab_size, char_emb_dim, word_emb_dim, hidden_size, out_size, dropout):
        """
        模型结构搭建
        :param vocab_size: 词表大小
        :param emb_dim: 词向量维数
        :param hidden_size: LSTM隐含层大小
        :param out_size: 标签数量
        """
        super(BiLSTM_CRF, self).__init__()
        # 定义字向量矩阵
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=char_emb_dim)

        self.dropout = nn.Dropout(dropout)

        input_size = char_emb_dim + word_emb_dim
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        # bilstm 输出将 hidden layer 的输出拼接
        self.hidden2tag = nn.Linear(in_features=hidden_size*2, out_features=out_size)

        self.crf = CRF(num_tags=out_size, batch_first=True)

    def forward(self, x):
        """
        :param x: (LongTensor: [batch, seq_len] , FloatTensor: [batch, seq_len, 300])
        :return:
        """
        char_input = x[0]
        word_embedding = x[1]
        char_embedding = self.embedding(char_input)
        embeddings = t.cat((char_embedding, word_embedding), dim=-1)
        output, (h_n, c_n) = self.bilstm(embeddings, None)
        output = self.dropout(output)
        output = self.hidden2tag(output)
        # crf 解码
        output = self.crf.decode(output)
        output = t.LongTensor(output)
        return output

    def log_likelihood(self, batch_x, batch_tag):
        """
        发射概率矩阵和真实标签的对数似然损失
        对数似然是负数，所以返回的时候要加个负号变成正数
        """
        char_input = batch_x[0]
        word_embedding = batch_x[1]
        char_embedding = self.embedding(char_input)
        # concat 字向量和词向量
        embeddings = t.cat((char_embedding, word_embedding), dim=-1)
        output, (h_n, c_n) = self.bilstm(embeddings, None)
        output = self.dropout(output)
        output = self.hidden2tag(output)
        return - self.crf(output, batch_tag)
