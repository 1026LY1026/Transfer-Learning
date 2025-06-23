import matplotlib.pyplot as plt
from IPython import display
import torch
import torch.nn as nn
import pandas as pd
import Anivisual as an
import numpy as np
from utility import *
import pyplot_make as pm
from torch.utils.data import DataLoader

log_test_acc = 0
log_train_acc = 0
log_train_rido = 0
log_net_str = " "
log_epoches = 0
log_lr = 0

attention_list = []


# The data loading function is responsible for loading data
# 数据加载函数，负责加载数据
def data_load(path, is_normalize, elements, key=False):
    Data_ = pd.read_csv(path)
    # Remove the first column, because the first column is a row mark, which is meaningless
    # 去掉第一列，因为第一列是行标，没有意义
    Data_.drop(elements, axis=1, inplace=True)
    # 是否做归一化
    for col_index, col_value in Data_.iteritems():
        minVals = col_value.min(0)
        maxVals = col_value.max(0)
        if maxVals - minVals > 1000:
            col_value = noramlization(col_value)
            Data_.loc[:, col_index] = np.array(col_value)

    # 取数据
    X_data = Data_.values.astype(np.float32)

    # 取目标值（或说取y 取标签，这里是rop）
    y_data = Data_.iloc[:, 4].values

    return X_data, y_data


# Make time series data set （It is divided into training set and test set）
# 制作成时间序列数据集(分为训练集和测试集)
def make_time_series_data(seq_length, pre_length, y_data, x_data, batch_size, is_shuffle=True):
    y_data = y_data[seq_length:]
    y_data = y_data.astype(np.float32)
    # an.log_train_rido = train_ratio

    # total data in torch format: input followed by output, non-uniform
    # 整理成torch  要求的格式 （[[数据]，[标签]]）
    data = []
    data_num = x_data.shape[0]
    for i in range(data_num - seq_length - pre_length):
        # input data
        input_temp = x_data[i:i + seq_length, :]
        # collect input and output
        data_temp = [input_temp, y_data[i:i + pre_length]]
        # append output column
        data.append(data_temp)
    # load data for iteration
    # shuffle = True: random load/sampling; =false: sequential load/sampling
    iter = DataLoader(data, batch_size=batch_size,
                                             shuffle=is_shuffle, num_workers=0)

    return iter

# 实时可视化训练和测试过程中的损失值和准确度
class Anivisual:
    def __init__(self, num_epochs):
        # 类的初始化方法，接受一个参数 num_epochs 表示总的训练轮次
        # 该方法设置了一些可视化的参数，并创建了两个子图用于显示损失值和准确度的变化
        from IPython import display
        from matplotlib.pyplot import MultipleLocator

        # 记录总的训练轮次
        self.num_epochs = num_epochs

        # 设置x,y轴主刻度间隔
        self.y_major_locator = MultipleLocator(0.3)
        self.x_major_locator = MultipleLocator(int(num_epochs / 5))

        # 创建包含两个子图的图形对象
        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 10))
        # 为两个子图设置网格
        self.axes[0].grid()
        self.axes[1].grid()

        # 设置子图的x轴范围
        self.axes[0].set_xlim([1, self.num_epochs])
        self.axes[1].set_xlim([1, self.num_epochs])
        # 设置子图的y轴范围
        self.axes[1].set_ylim([0, 3])

        # 设置x,y轴主刻度的位置
        self.axes[1].yaxis.set_major_locator(self.y_major_locator)
        self.axes[1].xaxis.set_major_locator(self.x_major_locator)

        # 记录训练和测试过程中的损失值和准确度
        self.train_loss_p = []
        self.test_loss_p = []
        self.train_p = []
        self.test_p = []

        # 记录当前的轮次和最大轮次
        self.p_i = []
        self.i = 1
        self.max_t = 0

    def pawn(self, train_loss, test_loss, train_, test_):
        # 类的方法，用于更新可视化图表并显示在当前环境中

        # 记录当前的训练轮次信息
        train_loss_legend = train_loss
        test_loss_legend = test_loss

        # 将当前轮次的信息添加到记录列表中
        self.train_loss_p.append(train_loss)
        self.test_loss_p.append(test_loss)
        self.train_p.append(train_)
        self.test_p.append(test_)
        self.p_i.append(self.i)

        # 构建图例的标签字符串
        str_train_loss_legend = "train_loss:" + str(round(train_loss_legend, 4))
        str_test_loss_legend = "test_loss:" + str(round(test_loss_legend, 4))
        str_train_ = "train_acc" + str(round(train_, 4))
        str_test_ = "test_acc" + str(round(test_, 4))

        # 在第一个子图中绘制训练和测试损失曲线
        self.axes[0].plot(self.p_i, self.train_loss_p, c='#448ee4', linestyle='-.', linewidth=2.50)
        self.axes[0].plot(self.p_i, self.test_loss_p, c='#e03fd8', linestyle='-.', linewidth=2.50)

        # 在第二个子图中绘制训练和测试准确度曲线
        self.axes[1].plot(self.p_i, self.train_p, c='#fb5581', linestyle='-.', linewidth=2.50)
        self.axes[1].plot(self.p_i, self.test_p, c='#56ae57', linestyle='-.', linewidth=2.50)

        # 设置图例并显示
        legend = [str_train_loss_legend, str_test_loss_legend, str_train_, str_test_]
        self.axes[0].legend(legend[0:2])
        self.axes[1].legend(legend[2:4])

        # 实时显示图表
        display.display(self.fig)
        display.clear_output(wait=True)
        self.i = self.i + 1

# 计算模型在给定数据迭代器中的准确度和均方误差（MSE）
def compute_acc(iter_, epoch, depth, pre_length, net, device, isTest=False):
    # 初始化数组用于存储预测值和真实值
    acc_ = np.zeros((1, pre_length))
    y_ = np.zeros((1, pre_length))

    # 遍历数据迭代器
    for X, y in iter(iter_):
        with torch.no_grad():
            # 将输入数据移动到设备上
            var_x = X.to(device)
            # 获取模型的预测值
            y_pre = net(var_x)
            # 将预测值和真实值存储到数组中
            acc_ = np.vstack((acc_, y_pre.cpu().detach().numpy()))
            y_ = np.vstack((y_, y.detach().numpy()))
    # 计算均方误差（MSE）
    mse_loss = ((acc_ - y_) ** 2).mean()
    # if epoch % 20 == 0 and isTest:
    #     # 出图
    #     rel_plt = []
    #     len_acc = acc_.shape[1]
    #     for item in range(len_acc):
    #         re = take_rela_error(y_[:, item], acc_[:, item])
    #         rel_plt.append(re)
    #     pm.distance_chart_plt(rel_plt, epoch, './png/2')
    #
    #     p_data = pd.concat([pd.DataFrame(y_), pd.DataFrame(acc_)], axis=1)
    #     p_data.drop(index=[0], inplace=True)
    #     p_data.insert(0, 'md', depth)
    #     p_data.to_csv('./data/F15a_by_F4F5F7F9F5_GRU_2.csv', sep=",", index=True)
    # 提取预测值和真实值的第一列
    acc_ = acc_[:, 0]
    y_ = y_[:, 0]

    # 去掉为0的项
    index_ = np.where(y_ == 0)
    y_ = np.delete(y_, index_)
    acc_ = np.delete(acc_, index_)

    # 计算准确度
    accuracy = 1 - (np.abs(acc_ - y_) / y_).mean()
    return accuracy, mse_loss

# 训练神经网络模型
def special_fit(net, epoches, train_iter, test_iter, lr, device_name, threshold_acc=1):
    """
    参数说明：
                net:网络模型
                epoches:训练的批次
                train_iter:训练集迭代器
                test_iter：测试集迭代器
                lr:学习率
                device_name:设备名称，是用cpu还是用gpu
                threshold_acc：精度上限，默认为1（测试集上的精度或者说预测能力）
                例子：special_fit(model,90,train_iter,test_iter,0.0001,"cuda")

    """
    # 将全局变量 y 定义为全局变量，用于记录日志
    global y

    # 模型结构，给日志
    # 设置设备（CPU 或 GPU）
    device = torch.device(device_name)
    # 记录模型结构到日志
    an.log_net_str = str(net)
    # 定义损失函数
    an.log_epoches = epoches
    an.log_lr = lr
    criterion = nn.MSELoss().to(device)
    # 定义优化器和参数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.001)  # weight_decay=0.001 范数，提高泛化能力
    # anma = Anivisual(epoches)
    # # 创建用于记录深度信息的数组
    md = []
    # 从测试集中获取深度信息
    for X, y in iter(test_iter):
        m = X[:, 0, 0]
        m = m.reshape(-1, 1)
        md.append(m)

    md = np.concatenate(md, axis=0)
    # 训练循环
    for epoch in range(epoches):
        # 将模型设置为训练模式
        net = net.train()
        net.to(device)
        index = 0
        # 遍历训练集迭代器
        for X, y in iter(train_iter):
            # get input 将数据和标签移到gpu
            var_x = X.to(device)
            var_y = y.to(device)
            # 将参数梯度归零
            optimizer.zero_grad()

            # 前向传播 + 反向传播 + 优化
            out = net(var_x)
            loss = criterion(out, var_y)
            loss.backward()
            optimizer.step()

            # 打印训练信息
            if index % 400 == 0:
                print('| epoch {:3d} | {:5d}/{:5d} index | '
                      'loss {:5.5f}'.format(
                    epoch, index, len(train_iter),
                    loss / len(train_iter)))  # , math.exp(cur_loss)

            index = index + 1
        # 计算准确度和均方误差
        pre_length = y.shape[1]
        net = net.eval()
        train_acc, train_loss = compute_acc(train_iter, epoch, md, pre_length=pre_length, net=net, device=device)
        test_acc, test_loss = compute_acc(test_iter, epoch, md, pre_length=pre_length, net=net, device=device, isTest=True)
        # anma.pawn(train_loss, test_loss, train_acc, test_acc)
        # 记录训练和测试的准确度到日志
        an.log_test_acc = test_acc
        an.log_train_acc = train_acc
        # 打印训练和测试的信息
        print('Epoch:', '%04d' % (epoch + 1), 'train_loss =', '{:.6f}'.format(train_loss), 'train_acc =',
              '{:.6f}'.format(train_acc), 'test_loss =', '{:.6f}'.format(test_loss), 'test_acc =',
              '{:.6f}'.format(test_acc))

        # 超过阈值，提前结束，与epoches配合使用，达到寻找最大精度的效果
        if test_acc > threshold_acc:
            break

# 使用训练好的神经网络模型进行预测
def predict(net, iter_, device_name):
    # 获取每个样本的特征数量和预测值的长度
    a, b = next(iter(iter_))[1].shape
    # 初始化用于存储预测值和真实值的数组
    y_pre = np.zeros((1, b))
    y_real = np.zeros((1, b))

    device = torch.device(device_name)
    net.to(device)
    net.eval()

    for x, y in iter_:
        with torch.no_grad():
            vail_x = x.to(device)
            # 获取模型的预测值
            y_test_pre = net(vail_x)
            # 将预测值和真实值存储到数组中
            y_pre = np.vstack((y_pre, y_test_pre.cpu().detach().numpy()))
            y_real = np.vstack((y_real, y.detach().numpy()))
    # 去掉第一行（初始化时的占位行）
    y_pre = y_pre[1:]
    y_real = y_real[1:]
    return y_pre, y_real

# 实验日志函数，打印实验日志
def experiment_log(experiment_name, ):
    import os
    # 检查是否存在log目录，如果不存在则创建
    isExists = os.path.exists("")
    if isExists == False:
        os.makedirs("")

    from datetime import datetime
    experiment_path = "/"
    # 构建实验日志的字符串
    str_w = "\n" + \
            "日志参数       |        参数值\n" + \
            "----------------------------------\n" + \
            "实验名称:" + "                  " + str(experiment_name) + "\n" + \
            "实验时间:" + "                  " + str(datetime.now()) + "\n" + \
            "测试集的精度：" + "               " + str(an.log_test_acc) + "\n" + \
            "训练集的精度：" + "               " + str(an.log_train_acc) + "\n" + \
            "训练集占比 ：" + "                " + str(an.log_train_rido) + "\n" + \
            "训练批次：" + "                  " + str(an.log_epoches) + "\n" + \
            "训练学习率：" + "                  " + str(an.log_lr) + "\n" + \
            "模型参数：：\n" + "                  " + log_net_str + "\n" + \
            "-------------------------------------------\n" + "\n\n"
    f1 = open(experiment_path + experiment_name + '.txt', 'a')
    f1.write(str_w)
    f1.close()

# 双向GRU（Gated Recurrent Unit）PyTorch模型类
class GRU_bi(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU_bi, self).__init__()
        # 初始化双向GRU层
        self.rnn1 = nn.GRU(
            input_size=input_size,
            hidden_size=int(hidden_size / 2),  # 由于是双向，所以隐藏层大小除以2
            num_layers=num_layers,
            bidirectional=True,  # 设置为双向
            batch_first=True  # 输入数据的维度顺序为 (batch_size, seq_length, input_size)
        )

    def forward(self, x):
        # 前向传播
        r_out, hn = self.rnn1(x, None)  # None 表示隐藏状态会使用全0的状态
        out = r_out

        return out

# 结合双向GRU和线性层的PyTorch模型类
class GRU_bi_toLinear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU_bi_toLinear, self).__init__()
        # 初始化双向GRU层
        self.rnn1 = nn.GRU(
            input_size=input_size,
            hidden_size=int(hidden_size / 2),  # 双向，所以隐藏层大小除以2
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        # 前向传播
        r_out, hn = self.rnn1(x, None)  # None 表示隐藏状态会使用全0的状态
        out = r_out[:, -1, :]  # 仅保留序列最后一个时间步的输出
        return out

# 单向GRU
class GRU_uni(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU_uni, self).__init__()
        # 初始化单向GRU层
        self.rnn1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,  # 设置为单向
            batch_first=True  # 输入数据的维度顺序为 (batch_size, seq_length, input_size)
        )

    def forward(self, x):
        # 前向传播
        r_out, hn = self.rnn1(x, None)  # None 表示隐藏状态会使用全0的状态
        out = r_out

        return out

    # Define GRU NETWORK structure

# 结合单向GRU和线性层的PyTorch模型类
class GRU_uni_toLinear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU_uni_toLinear, self).__init__()

        # 初始化单向GRU层
        self.rnn1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )

    def forward(self, x):
        # 前向传播
        r_out, hn = self.rnn1(x, None)  # None 表示隐藏状态会使用全0的状态
        out = r_out[:, -1, :]  # 仅保留序列最后一个时间步的输出

        return out

# 双向LSTM
class LSTM_bi(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM_bi, self).__init__()

        # 初始化双向LSTM层
        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=int(hidden_size / 2),  # 由于是双向，所以隐藏层大小除以2
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        # 前向传播
        r_out, hn = self.rnn1(x, None)  # None 表示隐藏状态会使用全0的状态
        out = r_out

        return out

    # Define GRU NETWORK structure

# 结合双向LSTM和线性层的PyTorch模型类
class LSTM_bi_toLinear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM_bi_toLinear, self).__init__()

        # 初始化双向LSTM层
        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=int(hidden_size / 2),  # 由于是双向，所以隐藏层大小除以2
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

    def forward(self, x):
        # 前向传播
        r_out, hn = self.rnn1(x, None)  # None 表示隐藏状态会使用全0的状态
        out = r_out[:, -1, :]  # 仅保留序列最后一个时间步的输出

        return out

# 单向LSTM
class LSTM_uni(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM_uni, self).__init__()

        # 初始化单向LSTM层
        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,  # 设置为单向
            batch_first=True  # 输入数据的维度顺序为 (batch_size, seq_length, input_size)
        )

    def forward(self, x):
        # 前向传播
        r_out, hn = self.rnn1(x, None)  # None 表示隐藏状态会使用全0的状态
        out = r_out

        return out

# 结合单向LSTM和线性层的PyTorch模型类
class LSTM_uni_toLinear(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM_uni_toLinear, self).__init__()

        # 初始化单向LSTM层
        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )

    def forward(self, x):
        # 前向传播
        r_out, hn = self.rnn1(x, None)  # None 表示隐藏状态会使用全0的状态
        out = r_out[:, -1, :]  # 仅保留序列最后一个时间步的输出

        return out

# 变分注意力层
# 作用：引入注意力机制，通过学习输入序列中每个时间步重要性，更好捕捉序列中关键信息
class var_attention_layer(nn.Module):
    def __init__(self, input_size):
        super(var_attention_layer, self).__init__()

        # 定义注意力层的结构
        self.atten = nn.Sequential(
            nn.Tanh(),  # Tanh激活函数
            nn.Linear(input_size, input_size),  # 线性层，用于映射特征
            nn.Softmax(2)  # 在第二个维度上进行Softmax操作，通常用于注意力权重
        )

        self.wei = None  # 用于存储计算得到的注意力权重

    def forward(self, x):
        # 计算注意力权重
        self.wei = self.atten(x).mean(0).mean(0)  # 求注意力矩阵，取均值操作
        # 利用注意力权重对输入进行加权
        x = torch.mul(x, self.wei)  # 注意力矩阵与特征矩阵相乘
        return x
