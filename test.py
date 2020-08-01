from main import predict
import json
import demjson
import re


def keywords_split(sentence: str):
    """切分keywords"""
    pattens = [r'[,，]', r'、', r'\s+', r'_', r'[|]']

    num_list = []
    for p in pattens:
        num_list.append(len(re.findall(p, sentence)))

    max_index = num_list.index(max(num_list))
    p = pattens[max_index]
    words = re.split(p, sentence.strip())
    return words


def main():
    with open(r'D:\Project File\数据融合\中国上市公司meta数据.jsonl', 'r', encoding='utf-8') as f:
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
            words = keywords_split(keywords)
            # predict(words)
            for word in words:
                if word != '':
                    tag_list, _ = predict(word)
                    if 'B-Com' and 'I-Com' in tag_list:
                        alias_list1.append(word)
        if description != '':
            _, alias_list2 = predict(description)

        alias_list = list(set(alias_list1 + alias_list2))
        for alias in alias_list:
            res['alias'].append(alias)

        with open('./alias.jsonl', 'a', encoding='utf-8') as fout:
            fout.write('{}\n'.format(json.dumps(res, ensure_ascii=False)))


main()
