from main import predict

keywords_path = r'C:\Users\suwen\Desktop\数据融合\7.18\keywords.txt'
with open(keywords_path, 'r', encoding='utf-8') as f:
    keywords = [line.strip() for line in f.readlines()]


fout_true = open('./true.txt', 'a', encoding='utf-8')
fout_false = open('./false.txt', 'a', encoding='utf-8')
for keyword in keywords:
    if keyword == '':
        print('空字符')
        continue
    try:
        tags = predict(keyword)
    except:
        tags = predict(keyword)
    if 'B-Com' in tags and 'I-Com' in tags:
        fout_true.write(keyword + '\n')
    else:
        fout_false.write(keyword + '\n')
