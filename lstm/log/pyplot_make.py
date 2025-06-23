import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from utility import *

# fig = plt.figure()
# fig.set_size_inches(10, 4)  # 整个绘图区域的宽度10和高度4

# ax = fig.add_subplot(1, 2, 1)
font = {'family': 'Times New Roman',
        'weight': '400',
        'size': 15,
        }


# Rop预测，并将实际值和预测值之间的关系以及拟合直线可视化保存为图像
def rop_prediction_plt(true, pre, epoch, path):
    # 创建线性回归模型
    regressor = LinearRegression()
    # 使用真实值和预测值训练线性回归模型
    regressor = regressor.fit(np.reshape(true, (-1, 1)), np.reshape(pre, (-1, 1)))
    # 打印拟合结果(参数)
    print(regressor.coef_, regressor.intercept_)
    # 画出数据和拟合直线的图
    plt.scatter(true, pre)  # 散点图
    plt.plot(np.reshape(true, (-1, 1)), regressor.predict(np.reshape(true, (-1, 1))), 'r')  # 拟合直线图
    plt.xlabel("actual value")  # x轴标签
    plt.ylabel("predictive value")  # y轴标签
    plt.title("Fitting results")  # 图标题
    # 保存图像
    plt.savefig(os.path.join(path, '%d.png' % epoch))
    # 关闭图形窗口，防止图像显示在Notebook中
    plt.close()

# 根据输入的相对误差数据和轮次，生成相对误差与距离元素之间的关系图
def distance_chart_plt(rel_size, epoch, path):

    unit = ["Relative_error", "Distance_length"]
    plt.figure(figsize=(24, 8))
    plt.plot(distance_elements, rel_size)
    # plt.plot(depth,col_one_pre)
    # plt.plot(depth,col_one_pre_attn)
    plt.legend(["ave_relative_error"], prop=font)
    plt.ylabel(unit[0], font)
    plt.xlabel(unit[1], font)
    # plt.tick_params(labelsize=20)
    # pyplot.savefig(os.path.join(png_save_path, 'pre.png'))
    plt.savefig(os.path.join(path, '%d.png' % epoch))
    plt.close()

# 根据输入的真实值、预测值、深度数据和轮次，生成真实值与预测值随深度变化的线性图
def line_chart_plt(true, pre, md, epoch, path):
    true = true.flatten()
    pre = pre.flatten()
    md = md.flatten()
    unit = ["ROP(m/hr)", "Depth(m)"]
    plt.figure(figsize=(24, 8))
    plt.plot(md, true)
    # plt.plot(depth,col_one_pre)
    # plt.plot(depth,col_one_pre_attn)
    plt.plot(md, pre)
    plt.legend(["real", "pre"], prop=font)
    plt.ylabel(unit[0], font)
    plt.xlabel(unit[1], font)
    plt.tick_params(labelsize=20)
    # pyplot.savefig(os.path.join(png_save_path, 'pre.png'))
    plt.savefig(os.path.join(path, '%d.png' % epoch))
    plt.close()


# 根据输入的相对误差数据、深度数据和轮次，生成相对误差与深度之间的散点图
def rel_error_plt(md, r, epoch, path):
    # 绘制相对误差与深度的散点图
    plt.scatter(r, md)
    # plt.plot(np.reshape(x, (-1, 1)), regressor.predict(np.reshape(x, (-1, 1))), 'r')
    plt.xlabel("relative_error")
    plt.ylabel("md")
    plt.title("Fitting results")
    plt.savefig(os.path.join(path, '%d.png' % epoch))
    # 关闭图形窗口
    plt.close()
