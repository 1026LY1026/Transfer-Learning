import torch.nn
# import numpy as np
import torch.nn as nn
from lstm.log import Anivisual as an
from torch.utils.data import DataLoader
from pyplot_make import *
from utility import *
# 引入画图的模块
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Parameter setting                                                参数设置
"""
 Parameter Description:                                            参数说明：
Path: the path where the dataset is located                             path：数据集所在的路径
seq_ Length: the length of the time slice                               seq_length：时间片的长度
pre_ Length: the predicted length of the model                          pre_length：模型预测长度
train_ Ratio: proportion of training set                                train_ratio：训练集占比
batch_ Size: batch size                                                 batch_size：批量大小
features_ Num: number of features                                       features_num：特征个数

"""
# 设置参数
# path = "./USROP_A 2 N-SH_F-14d.csv"
seq_length = 150 # 输入序列的长度，即模型将考虑的过去时间步的数量
pre_length = 50  # 预测序列的长度，即模型要预测的未来时间步的数量
# train_ratio = 0.7
batch_size = 64 # 每个训练批次中包含的样本数量
features_num = 12 # 每个时间步中的特征数量

# 导入数据集
x_data_4, y_data_4 = an.data_load(data_path_4, False, elements=elements)
# x_data_12, y_data_12 = an.data_load(data_path_12, False, elements=elements)
x_data_7, y_data_7 = an.data_load(data_path_7, False, elements=elements)
#x_data_9, y_data_9 = an.data_load(data_path_9, False, elements=elements)
x_data_5, y_data_5 = an.data_load(data_path_5, False, elements=elements)
# x_data_14, y_data_14 = an.data_load(data_path_14, False, elements=elements)
#x_data_15a, y_data_15a = an.data_load(data_path_15A, False, elements=elements, key=True)

# 两个数据集合并为训练集 然后制作训练集和测试机的时间序列数据集
train_x_data = np.vstack((x_data_4, x_data_5))
train_y_data = np.concatenate([y_data_4, y_data_5], axis=0)
train_data = an.make_time_series_data(seq_length, pre_length, train_y_data, train_x_data)
test_data = an.make_time_series_data(seq_length, pre_length, y_data_7, x_data_7)
print(f"train_data:{len(train_data)}, test_data:{len(test_data)}")

# load data for iteration 在每个训练迭代中加载新的数据，通过读取训练数据集的不同部分或随机抽样实现。
# DataLoader（）批量加载数据的工具类。将数据集划分成小批次，支持多线程异步加载数据以提高数据加载的效率。
# shuffle=True: 每个 epoch 开始时是否对数据进行洗牌（随机重排），洗牌可增加模型泛化能力，确保每个批次包含样本是随机选择而不是按原始顺序排列。
# num_workers=0: 指定数据加载时使用的子进程数量。设置为0表示在主进程中加载数据，而不使用额外的子进程。
train_iter = DataLoader(train_data, batch_size=batch_size,
                                         shuffle=True, num_workers=0)
test_iter = DataLoader(test_data, batch_size=batch_size,
                                        shuffle=True, num_workers=0)

# Define LSTM NETWORK structure
class lstm_uni_attention(nn.Module): # 定义一个继承自nn.Module的PyTorch模型类
    def __init__(self, input_size, hidden_size, num_layers, pre_length, seq_length):
        super(lstm_uni_attention, self).__init__()

        # 定义注意力机制，由一个Sequential层组成，包含Tanh激活函数、线性层和Softmax激活函数。目的：计算输入序列注意力权重
        self.atten = nn.Sequential(nn.Tanh(), nn.Linear(input_size, input_size), nn.Softmax(2))

        # 单向LSTM层
        self.rnn1 = nn.LSTM(
            input_size=input_size, # 包含输入大小
            hidden_size=hidden_size, # 隐藏层大小
            num_layers=num_layers, # LSTM层数
            bidirectional=False    # LSTM是否是双向
        )

        # 全连接层序列，包含多个线性层和ReLU激活函数。用于最终的输出，将LSTM的输出映射到预测序列的长度
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), int(hidden_size / 4)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 4), pre_length)
        )

    # 前向传播
    # r_out, hn: 。
    #
    # out: 将LSTM的输出通过全连接层序列self.fc映射到预测序列的长度，得到最终的输出。
    def forward(self, x):
        wei = self.atten(x).mean(0).mean(0)  # 求注意力矩阵，注意力权重矩阵
        x = torch.mul(x, wei)  # 注意力矩阵与特征矩阵相乘，应用注意力机制
        r_out, hn = self.rnn1(x, None)  # 使用定义的LSTM层进行前向传播，r_out是LSTM输出，hn是最后隐藏状态 None表示hidden state用全0的state
        out = self.fc(r_out[:, -1]) #  将LSTM输出通过全连接层序列self.fc映射到预测序列长度，得到最终输出
        return out


# 检查网络，定义网络的参数hidden_size和num_layers，并且查看网络结构，测试网络是否跑的通
# 定义网络的参数hidden_size和num_layers
hidden_size = 128
num_layers = 2
# 检查网络并且查看网络结构
lstm_uni_attention_test = lstm_uni_attention(input_size=features_num, hidden_size=hidden_size, num_layers=num_layers,
                                             pre_length=pre_length, seq_length=seq_length)
print(lstm_uni_attention_test, lstm_uni_attention_test(torch.randn(batch_size, seq_length, features_num)).shape)

# 模型训练
device = torch.device("cpu")
net = lstm_uni_attention_test
print(net)
epoches = 20
lr = 0.00015
# threshold_acc=1: 可选参数，默认值为 1。表示训练过程中达到的准确度阈值。模型在验证集的准确度超过阈值时训练可能会提前停止
an.special_fit(net, epoches, train_iter, test_iter, lr, device, threshold_acc=1)

# 使用日志，进行实验日志记录
experiment_name = "gru_uni单循环加注意力机制"
# experiment_path=" "
an.experiment_log(experiment_name)

# # 测试画图
# train_ratio = 0
# batch_size = 128

real_test_iter = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

# net.eval() PyTorch 中用于将神经网络模型切换到评估（evaluation）模式 net.train()切换回训练模式
# 在训练时，Batch Normalization 层和 Dropout 层通常会考虑整个批次的统计信息，以及在每个前向传播中随机丢弃一些神经元。
# 在评估模式下，Batch Normalization 会使用之前训练时计算的统计信息，而 Dropout 不再随机丢弃神经元，而是保留所有神经元。
# 梯度计算和参数更新：评估模式下，PyTorch 不会追踪梯度，在测试阶段不需要进行反向传播和参数更新。
# 训练模式，模型通常需要追踪梯度以便进行反向传播和参数更新。
net = net.eval()

y_pre = np.zeros((1, pre_length))  #  创建一个初始大小为 (1, pre_length) 的全零数组，存储模型预测结果
y_real = np.zeros((1, pre_length)) # 存储真实标签（实际值）
net.to(device)
for x, y in real_test_iter:
    with torch.no_grad(): # 上下文管理器，确保在下面的代码块中不计算梯度。测试阶段，通常不需要计算梯度。
        vail_x = x.to(device)
        y_test_pre = net(vail_x)

        # 将每个样本的预测/真实结果堆叠到y_pre数组。
        y_pre = np.vstack((y_pre, y_test_pre.cpu().detach().numpy()))
        y_real = np.vstack((y_real, y.detach().numpy()))
print(y_real.shape, y_pre.shape)

print(type(y_real))
print(type(y_pre))

# # 对数组 y_pre 进行切片操作，将其划分为三个部分 cbar_rop_left、cbar_rop_center 和 cbar_rop_right，并根据相应的索引范围提取数据
# cbar_rop_left = y_pre[0:pre_length * 3]
# cbar_rop_center = y_pre[int(y_pre.shape[0] / 2):int(y_pre.shape[0] / 2) + pre_length * 3]
# cbar_rop_right = y_pre[y_pre.shape[0] - pre_length * 3:y_pre.shape[0]]

# plt.figure(figsize=(10, 10))
# ax1 = plt.subplot(1, 4, 1)  # 要生成两行两列，这是第一个图plt.subplot('行','列','编号')
# ax1.imshow(cbar_rop_left)
# ax2 = plt.subplot(1, 4, 2)  # 两行两列,这是第二个图
# ax2.imshow(cbar_rop_center)
# plt.yticks([])  # 去y坐标刻度
# ax3 = plt.subplot(1, 4, 3)  # 两行两列,这是第三个图
# im = ax3.imshow(cbar_rop_right)
# plt.yticks([])  # 去y坐标刻度
# plt.subplots_adjust(wspace=0.03) # 调整子图之间的水平间距
# plt.colorbar(im, ax=[ax1, ax2, ax3], fraction=0.04, pad=0.03) # 在图表右侧添加颜色条，fraction和pad参数控制颜色条位置和大小
# plt.savefig('./png/1/lstm_therm.png')
# plt.close()

# depth = x_data_7[seq_length:-pre_length + 1, 0] # 测试集

# col_one_real = y_real[:, 0].copy()
# col_one_pre = y_pre[:, 0].copy()
# # Prapare font for drawing
# font = {'family': 'Times New Roman',
#         'weight': '400',
#         'size': 25,
#         }
# unit = ["ROP, m/hr", "Depth, m"]
# plt.figure(figsize=(24, 8))
# plt.plot(depth, col_one_real)
# # plt.plot(depth,col_one_pre)
# # plt.plot(depth,col_one_pre_attn)
# plt.plot(depth, col_one_pre)
# plt.legend(["real", "col_one_pre_single"], prop=font)
# plt.ylabel(unit[0], font)
# plt.xlabel(unit[1], font)
# plt.tick_params(labelsize=20)
# plt.savefig('./png/1/lstm_ROP_Depth.png')
# plt.close()

# r=mean_relative_error(col_one_real, col_one_real)
# path='./png/2/rop_prediction_plt.png'
# rop_prediction_plt(col_one_real,col_one_real,epoches,path)
# rel_error_plt(depth, r, epoches, path)
# # distance_chart_plt(rel_size, epoches, path)
# line_chart_plt(col_one_real, col_one_real, depth, epoches, path)