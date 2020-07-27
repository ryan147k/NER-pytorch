from main import predict
from utils import load_word_embedding
import json
import demjson


word_emb = load_word_embedding()


with open('./中国上市公司meta数据.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    dic = demjson.decode(line)
    name = dic['compName']
    keywords = dic['keywords']
    description = dic['description']

    res = {
        "name": name,
        "alias": []
    }

    alias_list1, alias_list2 = [], []
    if keywords != '':
        _, alias_list1 = predict(keywords, word_emb)
    if description != '':
        _, alias_list2 = predict(description, word_emb)

    alias_list = list(set(alias_list1 + alias_list2))
    for alias in alias_list:
        res['alias'].append(alias)

    with open('./alias.josnl', 'a', encoding='utf-8') as fout:
        fout.write('{}\n'.format(json.dumps(res, ensure_ascii=False)))
