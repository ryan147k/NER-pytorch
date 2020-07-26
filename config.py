# coding=utf-8
import warnings

import os
import sys
ROOT = os.path.dirname(__file__)
sys.path.append(ROOT)


class DefaultConfig:
    max_length = 88  # 数据中一个句子的最大长度
    vocab_size = 2350   # 字表大小
    emb_dim = 100    # 字向量维数
    tag_num = 3     # 标签数量

    max_epoch = 30
    lr = 1e-3
    dropout = 0.5
    lr_decay = 0.95
    weight_decay = 1e-4

    vocab_file_path = os.path.join(ROOT, './resource/rmrb/vocab.json')
    tag_file_path = os.path.join(ROOT, './resource/rmrb/tag.json')
    train_data_path = os.path.join(ROOT, './resource/rmrb/train.txt')
    test_data_path = os.path.join(ROOT, './resource/rmrb/test.txt')
    load_model_path = os.path.join(ROOT, './ckpts/BiLSTM_CRF_0726_12h39m48s.pth')
    # load_model_path = None


def parse(self, kwargs):
    """
    根据字典kwargs 更新 config参数
    """
    # 更新配置参数
    for k, v in kwargs.items():
        if not hasattr(self, k):
            # 警告还是报错，取决于你个人的喜好
            warnings.warn("Warning: opt has not attribut %s" % k)
        setattr(self, k, v)

    # 打印配置信息
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            attr = getattr(self, k)
            for _ in range(20 - len(k)):
                k += ' '
            print(f"{k}\t{attr}")


DefaultConfig.parse = parse
opt = DefaultConfig()
