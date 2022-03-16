# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse


#声明argparse对象 可附加说明
parser = argparse.ArgumentParser(description='Chinese Text Classification')

#添加参数
#模型是必须设置的参数(required=True) 类型是字符串
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN、TextRCNN, TextRNN_Att')
#embedding随机初始化或使用预训练词或字向量 默认使用预训练
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
#基于词还是基于字 默认基于字
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')

#解析参数
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'

    #获取选择的模型名字
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN,TextRCNN, TextRNN_Att

    from utils import build_dataset, build_iterator, get_time_dif

    # 根据所选模型名字在models包下 获取相应模块(.py)
    x = import_module('models.' + model_name)
    # 每一个模块(.py)中都有一个模型定义类和与该模型相关的配置类(定义该模型的超参数) 初始化配置类的对象
    config = x.Config(dataset, embedding)


    #设置随机种子 确保每次运行的条件(模型参数初始化、数据集的切分或打乱等)是一样的
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # 数据预处理
    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)

    # 构建训练集、验证集、测试集迭代器/生成器（节约内存、避免溢出）
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    # 构造模型对象
    config.n_vocab = len(vocab) #词典大小可能不确定，在运行时赋值
    # 构建模型对象 并to_device
    model = x.Model(config).to(config.device)
    #使用自定义的参数初始化方式
    init_network(model)
    print(model.parameters)

    #训练、验证和测试
    train(config, model, train_iter, dev_iter, test_iter)
