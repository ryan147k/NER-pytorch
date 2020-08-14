# companyner使用方法
#### 加载词向量
` cn.utils.load_word_embedding(word_emb_path)`

word_emb_path: 二进制词向量文件路径

### keywords 格式切分

eg.  海天,海天酱油,海天黄豆酱 -> [海天, 海天酱油, 海天黄豆酱]

以出现最多的分隔符切分（空格，逗号，顿号，|）

` cn.keywords_split(keywords)`

### 公司名识别

`cn.predict(sentence, word_emb)`

sentence: 待识别的句子

word_emb: 加载完成的词向量对象

