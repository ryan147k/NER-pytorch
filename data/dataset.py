# coding=utf-8
import torch as t
from torch.utils.data import Dataset
import json
from config import opt
from utils import word2idx, tag2idx
import jieba


class RmrbDataset(Dataset):
    """
    人民日报数据集
    """
    def __init__(self, word_emb, train=True, validate=False, test_sentence=None):
        super(RmrbDataset, self).__init__()
        # 读取词表字典
        vocab_dic = json.loads(
            open(opt.vocab_file_path, 'r', encoding='utf-8').read()
        )
        # 读取标签字典
        tag_dic = json.loads(
            open(opt.tag_file_path, 'r', encoding='utf-8').read()
        )
        # 构造数据集
        self.X_data, self.Y_data = [], []
        if train:
            fin = open(opt.train_data_path, 'r', encoding='utf-8')
            data = fin.read().strip()     # 去除头尾回车
        elif validate:
            fin = open(opt.test_data_path, 'r', encoding='utf-8')
            data = fin.read().strip()     # 去除头尾回车
        else:
            assert test_sentence is not None
            data = ''
            for c in test_sentence:
                data += '{} O\n'.format(c)
            data = data.strip()

        sentences = data.split('\n\n')        # 切分句子
        for s in sentences:
            x, y = [], []
            items = s.split('\n')
            seq_len = len(items)        # 句子长度
            # 切分词和标签
            for item in items:
                features = item.split()
                character = features[0]
                tag = features[-1]
                x.append(character)
                y.append(tag)

            # 加入词向量
            word_sequence = self._word_embedding(x, word_emb)

            x = word2idx(x)
            y = tag2idx(y)
            # 句子补齐和截断
            if seq_len > opt.max_length:
                x = x[:opt.max_length]
                y = y[:opt.max_length]
            else:
                for _ in range(opt.max_length - seq_len):
                    x.append(vocab_dic['_pad'])
                    y.append(tag_dic['O'])
            x = t.LongTensor(x)
            y = t.LongTensor(y)
            self.X_data.append((x, word_sequence))
            self.Y_data.append(y)

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index]

    def __len__(self):
        return len(self.X_data)

    def _word_embedding(self, char_list, word_emb):
        """
        加载预训练好的词向量
        """
        t.manual_seed(2)
        # np.random.seed(2)
        emb_unk = t.randn(300)

        sequence = t.Tensor()
        sentence = ''
        for c in char_list:
            sentence += c
        words = jieba.lcut(sentence)
        for word in words:
            try:
                embedding = t.from_numpy(word_emb[word]).float()
            except KeyError:
                embedding = emb_unk
            embedding = embedding.unsqueeze(dim=0)
            # print(embedding.shape)
            for c in word:
                sequence = t.cat((sequence, embedding), dim=0)
        # 截断
        if len(sequence) > opt.max_length:
            sequence = sequence[:opt.max_length]
        # 补齐
        for _ in range(opt.max_length - len(sentence)):
            sequence = t.cat((sequence, emb_unk.unsqueeze(dim=0)), dim=0)
        return sequence


if __name__ == '__main__':
    pass
