# coding=utf-8
import torch as t
from model import BiLSTM_CRF
from config import opt
from data import RmrbDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tensorboardX import SummaryWriter
from utils import idx2tag
from utils import load_word_embedding


def train():
    """
    模型训练
    """
    train_writer = SummaryWriter(log_dir='./log/train')
    test_writer = SummaryWriter(log_dir='./log/test')

    # step1 模型
    bilstm_crf = BiLSTM_CRF(opt.vocab_size, opt.char_emb_dim, opt.word_emb_dim,
                            (opt.char_emb_dim+opt.word_emb_dim)//2,
                            opt.tag_num, dropout=opt.dropout)
    if opt.load_model_path:     # 是否加载checkpoint
        bilstm_crf.load(opt.load_model_path)

    # step2 数据
    word_emb = load_word_embedding()
    rmrb_train_dataset = RmrbDataset(word_emb=word_emb, train=True)
    rmrb_test_dataset = RmrbDataset(word_emb=word_emb, train=False)
    rmrb_train_dataloader = DataLoader(rmrb_train_dataset, batch_size=64, shuffle=True)
    rmrb_test_dataloader = DataLoader(rmrb_test_dataset, batch_size=len(rmrb_test_dataset), shuffle=True)

    # step3 损失函数和优化器
    # loss_fn = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(params=bilstm_crf.parameters(), lr=lr, weight_decay=opt.weight_decay)

    previous_loss = 1e9
    iteration = 0
    for epoch in range(opt.max_epoch):
        print('epoch {}'.format(epoch))
        for ii, (x_batch, y_batch) in enumerate(rmrb_train_dataloader):
            # 计算loss
            loss = bilstm_crf.log_likelihood(x_batch, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if ii % 20 == 0:
                # print('loss:{}'.format(loss.item()))
                train_writer.add_scalar('Loss', loss.item(), iteration)
                iteration += 1
                if loss > previous_loss:
                    lr = lr * opt.lr_decay
                else:
                    previous_loss = loss.item()
        # 保存模型检查点
        bilstm_crf.save()

        # 评价指标
        with t.no_grad():
            bilstm_crf.eval()   # 将模型设置为验证模式
            for x_test, y_test in rmrb_test_dataloader:
                test_loss = bilstm_crf.log_likelihood(x_test, y_test)
                test_writer.add_scalar('Loss', test_loss.item(), iteration)
                y_pre = bilstm_crf(x_test)
                print(classification_report(t.flatten(y_test), t.flatten(y_pre)))
            bilstm_crf.train()  # 将模型恢复成训练模式


def validate():
    """
    模型验证
    """
    # 模型
    bilstm_crf = BiLSTM_CRF(opt.vocab_size, opt.char_emb_dim, opt.word_emb_dim,
                            (opt.char_emb_dim+opt.word_emb_dim)//2,
                            opt.tag_num, dropout=opt.dropout)
    if opt.load_model_path:
        bilstm_crf.load(opt.load_model_path)

    # 数据
    word_emb = load_word_embedding()
    validate_dataset = RmrbDataset(word_emb, train=False, validate=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=len(validate_dataset))
    for i, (x_batch, y_batch) in enumerate(validate_dataloader):
        y_hat = bilstm_crf(x_batch)
        print(classification_report(t.flatten(y_batch), t.flatten(y_hat)))


def predict(sentence, print_entity=False):
    """
    模型预测
    """
    # 模型
    bilstm_crf = BiLSTM_CRF(opt.vocab_size, opt.char_emb_dim, opt.word_emb_dim,
                            (opt.char_emb_dim+opt.word_emb_dim)//2,
                            opt.tag_num, dropout=opt.dropout)
    if opt.load_model_path:     # 是否加载checkpoint
        bilstm_crf.load(opt.load_model_path)

    # 数据
    word_emb = load_word_embedding()
    test_dataset = RmrbDataset(word_emb, train=False, validate=False, test_sentence=sentence)

    x = list(test_dataset[0][0])  # 拿到该句子的input 表示
    # TODO: dataloader的batch是将x里面的每个tensor都增加一个维度
    x[0] = x[0].unsqueeze(dim=0)
    x[1] = x[1].unsqueeze(dim=0)

    tag_idx = bilstm_crf(x).squeeze()
    tag_idx = tag_idx.numpy().tolist()

    if print_entity:
        entity_list = []
        i = 0
        while i < len(tag_idx):
            if tag_idx[i] == 1:
                entity = sentence[i]
                for j in range(i+1, len(tag_idx)):
                    if tag_idx[j] == 2:
                        i = j + 1
                        entity += sentence[j]
                    else:
                        i = j
                        break
                entity_list.append(entity)
            else:
                i += 1
        print(entity_list)
        print('\n')

    return idx2tag(tag_idx)


if __name__ == "__main__":
    print(predict("中石化、中石油和中国建筑(5.030, -0.03, -0.59%)进入榜单前三名。中国平安(75.940, -1.06, -1.38%)、工行银行、中铁股份、上汽集团(17.970, 0.01, 0.06%)、中国铁建(8.730, -0.08, -0.91%)、中国移动、中国人寿(35.020, -0.38, -1.07%)进入前十。", print_entity=True))
    # train()
    # validate()
