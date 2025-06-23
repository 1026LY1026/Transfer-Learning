# 导入必要的模块
# 绘图与数据分析模块
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utility import *
from pylab import mpl

# 设置matplotlib的配置
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 忽略警告提示
import warnings

warnings.filterwarnings('ignore')


def drwa(path,name,out_path):
    data=pd.read_csv(path)
    rop=data.iloc[:,-1]
    plt.figure(figsize=(10, 5))
    plt.plot(rop, label=name, color='blue', linewidth=3)
    plt.grid()
    plt.savefig(out_path)
    plt.legend()
    plt.show()



data_path_bz_1='tf/train/data/bz/BZ19_6_1.csv'
data_path_bz_2='tf/train/data/bz/BZ19_6_2.csv'
data_path_bz_3='tf/train/data/bz/BZ19_6_3.csv'
data_path_bz_4='tf/train/data/bz/BZ19_6_4.csv'
data_path_bz_5='tf/train/data/bz/BZ19_6_5.csv'
data_path_bz_6='tf/train/data/bz/BZ19_6_6.csv'
data_path_bz_7='tf/train/data/bz/BZ19_6_7.csv'
data_path_bz_8='tf/train/data/bz/BZ19_6_8.csv'
data_path_bz_9='tf/train/data/bz/BZ19_6_9.csv'
data_path_bz_10='tf/train/data/bz/BZ19_6_10.csv'
data_path_bz_11='tf/train/data/bz/BZ19_6_11.csv'
data_path_bz_12='tf/train/data/bz/BZ19_6_12.csv'
data_path_bz_14='tf/train/data/bz/BZ19_6_14.csv'
data_path_bz_15='tf/train/data/bz/BZ19_6_15.csv'
data_path_bz_16='tf/train/data/bz/BZ19_6_16.csv'


drwa(data_path_bz_4,'bz_4','tf/train/rop/bz_4.png')