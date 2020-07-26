# coding=utf-8
import torch as t
from torch.utils.data import Dataset
import json
from config import opt
from utils import word2idx, tag2idx


class RmrbDataset(Dataset):
    """
    人民日报数据集
    """
    def __init__(self, train=True):
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
        else:
            fin = open(opt.test_data_path, 'r', encoding='utf-8')

        data = fin.read().strip()     # 去除头尾回车
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
            self.X_data.append(x)
            self.Y_data.append(y)
        self.X_data = t.LongTensor(self.X_data)
        self.Y_data = t.LongTensor(self.Y_data)

    def __getitem__(self, index):
        return self.X_data[index], self.Y_data[index]

    def __len__(self):
        return len(self.X_data)


if __name__ == '__main__':
    pass
