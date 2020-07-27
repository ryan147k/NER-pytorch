# coding=utf-8
from config import opt
import json
import time
from gensim.models import KeyedVectors


def word2idx(word_list):
    """
    句子转成数字列表
    """
    vocab_dic = json.loads(
        open(opt.vocab_file_path, 'r', encoding='utf-8').read()
    )

    idx = []
    for word in word_list:
        if word.encode('utf-8').isalpha():
            word = '_letter'
        elif word.isdigit():
            word = '_num'
        elif word not in vocab_dic.keys():
            word = '_unk'
        idx.append(vocab_dic[word])
    return idx


def tag2idx(tag_list):
    """
    标签转成数字列表
    """
    tag_dic = json.loads(
        open(opt.tag_file_path, 'r', encoding='utf-8').read()
    )

    idx = []
    for tag in tag_list:
        idx.append(tag_dic[tag])
    return idx


def idx2tag(idx):
    """
    数字列表转为标签列表
    """
    assert type(idx) == list
    tag_dic = json.loads(
        open(opt.tag_file_path, 'r', encoding='utf-8').read()
    )
    tag_dic_reverse = {v: k for k, v in tag_dic.items()}

    tag_list = []
    for i in idx:
        tag_list.append(tag_dic_reverse[i])
    return tag_list


def load_word_embedding():
    """
    加载词向量到内存中
    """
    print("Loading word embedding, Please wait")
    start = time.time()
    word_emb = KeyedVectors.load_word2vec_format(opt.word_emb_path)
    cost = time.time() - start
    print("Load complete, take {}min{}sec".format(int(cost/60), int(cost % 60)))
    return word_emb
