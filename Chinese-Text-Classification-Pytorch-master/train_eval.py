# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    # 训练模式
    model.train()

    # 学习率指数衰减，每过一个epoch：学习率 = gamma * 学习率 也可以采用达观杯中的方式，当模型在验证集上指标不再提高时，学习率减半
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)#前向传播,获取输出
            model.zero_grad()#清空梯度
            # 计算交叉熵损失（内部包含softmax log等操作） 可以用nn里面的函数 也可以用F中的函数 labels为整数索引 内部会自动转换为one-hot
            loss = F.cross_entropy(outputs, labels)
            loss.backward()#计算梯度
            optimizer.step()#更新参数
            # 每100个batch 计算一下在验证集上的指标
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                # 当前batch上训练集的准确率 因为是类别均衡数据集，所以可以直接用准确率作评估指标
                train_acc = metrics.accuracy_score(true, predic)
                # 计算此时模型在验证集上的损失和准确率
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                # 保存在验证集上损失最小的参数
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # 保存 训练集（当前batch）、验证集的损失和准确率信息 方便可视化 以batch为单位
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    # 模型训练结束后 进行测试
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    # 加载使验证集损失最小的参数
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    # 计算测试集准确率，每个batch的平均损失 分类报告、混淆矩阵
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    # 存储验证集所有batch的预测结果
    predict_all = np.array([], dtype=int)
    # 存储验证集所有batch的真实标签
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    # 计算验证集准确率
    acc = metrics.accuracy_score(labels_all, predict_all)
    # 如果是测试集的话 计算一下分类报告和混淆矩阵
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        # 计算混淆矩阵
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    # 返回准确率和每个batch的平均损失
    return acc, loss_total / len(data_iter)