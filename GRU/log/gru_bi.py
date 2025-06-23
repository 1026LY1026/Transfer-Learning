import torch
import torch.nn
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import Anivisual as an
from utility import *
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 数据集路径
# path = "../first_test/data/USROP_A 2 N-SH_F-14d.csv"

# 时间片的长度
seq_length = 100
# 模型预测长度
pre_length = 50
# 批量大小
batch_size = 64
# 特征个数
input_size = 12
# 是否随机抽样
is_shuffle = True
# 训练批次
epoches = 50

x_data_4, y_data_4 = an.data_load(data_path_4, False, elements=elements)
# x_data_12, y_data_12 = an.data_load(data_path_12, False, elements=elements)
x_data_7, y_data_7 = an.data_load(data_path_7, False, elements=elements)
# x_data_9, y_data_9 = an.data_load(data_path_9, False, elements=elements)
x_data_5, y_data_5 = an.data_load(data_path_5, False, elements=elements)
# x_data_14, y_data_14 = an.data_load(data_path_14, False, elements=elements)
train_x_data = np.vstack((x_data_4, x_data_5))
train_y_data = np.concatenate([y_data_4, y_data_5], axis=0)
# x_data_9a, y_data_9a = an.data_load(data_path_9A, False, elements=elements)
# x_data_15a, y_data_15a = an.data_load(data_path_15A, False, elements=elements)

train_iter = an.make_time_series_data(seq_length, pre_length, train_y_data, train_x_data, batch_size)
test_iter = an.make_time_series_data(seq_length, pre_length, y_data_7, x_data_7, batch_size)
print(f"'训练集总样本数量':{len(train_iter) * batch_size},'测试集总样本数量':{len(test_iter) * batch_size}")

# 模型搭建，与模型结构测试
# 创建一个顺序容器
model = nn.Sequential()
# 添加第一层GRU（双向）
model.add_module('1', an.GRU_bi(input_size, 128, 1))
# 可以在外部添加更多的gru，达到逐层递减的采样效果 第二层（结合双向GRU和线性层的PyTorch模型类）
model.add_module('2', an.GRU_bi_toLinear(128, 64, 1))
# 添加一个线性层，完成与输出层的连接，内部也是一个GRU层
model.add_module('3', nn.Linear(64, 32))
# 添加ReLU激活函数
model.add_module('4', nn.ReLU())
# 添加输出层，输出长度为pre_length
model.add_module('5', nn.Linear(32, pre_length))
# 打印模型结构和输入的输出形状
print(f"模型结构: {model}, 输入的输出形状: {model(torch.randn(batch_size, seq_length, input_size)).shape}")
# 使用an.special_fit函数对模型进行训练
an.special_fit(model, epoches, train_iter, test_iter, 0.0001, "cpu")

# 测试集预测
# 重新设置批量大小
batch_size = 128  # 批量大小尽可能大
# train_iter = an.make_time_series_data(seq_length, pre_length, train_y_data, train_x_data, batch_size)
# 生成测试集的迭代器（注意设置is_shuffle=False，避免重新洗牌）
test_iter = an.make_time_series_data(seq_length, pre_length, y_data_7, x_data_7, batch_size, is_shuffle=False)
# 使用an.predict函数进行模型在测试集上的预测
y_pre, y_real = an.predict(model, test_iter, "cpu")
# 打印预测结果的形状
print(y_pre.shape, y_real.shape)

# 使用日志，进行实验日志记录
experiment_name = "gru_bi双循环"
# experiment_path=" "
an.experiment_log(experiment_name)

# 从预测结果中提取用于绘制热图的数据
cbar_rop_left = y_pre[0:pre_length * 3]
cbar_rop_center = y_pre[int(y_pre.shape[0] / 2):int(y_pre.shape[0] / 2) + pre_length * 3]
cbar_rop_right = y_pre[y_pre.shape[0] - pre_length * 3:y_pre.shape[0]]

# 创建一个10x10的图表
plt.figure(figsize=(10, 10))

# 分别在图表的四个位置绘制三个热图
ax1 = plt.subplot(1, 4, 1)
ax1.imshow(cbar_rop_left)
ax2 = plt.subplot(1, 4, 2)
ax2.imshow(cbar_rop_center)
plt.yticks([])
ax3 = plt.subplot(1, 4, 3)
im = ax3.imshow(cbar_rop_right)
plt.yticks([])

# 设置子图之间的空白，并在最后一个子图上添加颜色条
plt.subplots_adjust(wspace=0.03)
plt.colorbar(im, ax=[ax1, ax2, ax3], fraction=0.04, pad=0.03)

# 保存热图为PNG文件
plt.savefig('./png/gru_therm.png')
plt.close()

# 获取深度数据
depth = x_data_7[seq_length:-pre_length, 0]

# 打印深度数据的形状
print(depth.shape)

# 打印预测结果和深度数据的形状
print(y_real.shape, depth.shape)

# 提取真实值和预测值的第一列数据
col_one_real = y_real[:, 0].copy()
col_one_pre = y_pre[:, 0].copy()

# 准备用于绘制的字体
font = {'family': 'Times New Roman',
        'weight': '400',
        'size': 25,
        }

# 创建一个24x8的图表，并在图表上绘制真实值和预测值的深度-ROP图
plt.figure(figsize=(24, 8))
plt.plot(depth, col_one_real)
plt.plot(depth, col_one_pre)
plt.legend(["col_one_real", "col_one_pre"], prop=font)
plt.ylabel("ROP m/hr", font)
plt.xlabel("Depth m", font)
plt.tick_params(labelsize=20)

# 保存深度-ROP图为PNG文件
plt.savefig('./png/gru_ROP_Depth.png')
plt.close()


# 定义一个注意力层，作用是计算输入数据注意力矩阵，使用该注意力矩阵对输入数据进行加权
class attention_layer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pre_length, seq_length):
        super(attention_layer, self).__init__()
        # 定义注意力机制的顺序容器，包括Tanh激活函数、全连接层、Softmax激活函数
        self.atten = nn.Sequential(nn.Tanh(), nn.Linear(input_size, input_size), nn.Softmax(2))

    def forward(self, x):
        # 计算注意力权重，对输入数据进行加权
        wei = self.atten(x).mean(0).mean(0)  # 求注意力矩阵
        x = torch.mul(x, wei)  # 注意力矩阵与特征矩阵相乘
        return x

