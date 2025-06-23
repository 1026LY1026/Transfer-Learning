import numpy as np
import pandas as pd

data_path_4 = './data/volve/S_F4.csv'
data_path_5 = './data/volve/S_F5.csv'
data_path_7 = './data/volve/S_F7.csv'
data_path_9 = './data/volve/S_F9.csv'
data_path_9A = './data/volve/S_F9A.csv'
data_path_10 = './data/volve/S_F10.csv'
data_path_12 = './data/volve/S_F12.csv'
data_path_14 = './data/volve/S_F14.csv'
data_path_15A = './data/volve/S_F15A.csv'

data_path_well_1 = './data/XJ_well/well_1.csv'
data_path_well_2 = './data/XJ_well/well_2.csv'
data_path_well_3 = './data/XJ_well/well_3.csv'
data_path_well_4 = './data/XJ_well/well_4.csv'


data_path_bz_1='./data/bz/BZ19_6_1.csv'
data_path_bz_2='./data/bz/BZ19_6_2.csv'
data_path_bz_3='./data/bz/BZ19_6_3.csv'
data_path_bz_4='./data/bz/BZ19_6_4.csv'
data_path_bz_5='./data/bz/BZ19_6_5.csv'
data_path_bz_6='./data/bz/BZ19_6_6.csv'
data_path_bz_7='./data/bz/BZ19_6_7.csv'
data_path_bz_8='./data/bz/BZ19_6_8.csv'
data_path_bz_9='./data/bz/BZ19_6_9.csv'
data_path_bz_10='./data/bz/BZ19_6_10.csv'
data_path_bz_11='./data/bz/BZ19_6_11.csv'
data_path_bz_12='./data/bz/BZ19_6_12.csv'
data_path_bz_14='./data/bz/BZ19_6_14.csv'
data_path_bz_15='./data/bz/BZ19_6_15.csv'
data_path_bz_16='./data/bz/BZ19_6_16.csv'


# max_min_distance = 500
# train_batch = 256
# test_batch =256
# pre_len = 50
# seq_len = 300
# interval = 20
# tf_lr = 0.00005    # 0.0005 0.00005 0.00002
# times = 100
# best_acc = 0.99

train_batch = 64
test_batch =64
pre_len = 50
seq_len = 300
interval = 20
tf_lr = 2e-4# 2e-4
times = 100
best_acc = 0.99



def extract_anti_diagonal_blocks(matrix):
    rows, cols = matrix.shape
    blocks = []

    # 分割所有的数据块 不足的补0
    sub_blocks = []
    for start_row in range(0, rows, interval):
        for col in range(cols):
            block = []
            for i in range(interval):
                if start_row + i < rows:
                    block.append(matrix[start_row + i, col])
                else:
                    block.append(-999)  # 填充0
            sub_blocks.append((start_row // interval, col, block))

    # 获取所有反对角线的位置
    anti_diagonals = {}
    for r, c, block in sub_blocks:
        if (r + c) not in anti_diagonals:
            anti_diagonals[(r + c)] = []
        anti_diagonals[(r + c)].append(block)

    # 转换反对角线变成数据块
    for key in sorted(anti_diagonals.keys()):
        blocks.append(anti_diagonals[key])

    result = []
    for arr in blocks:
        if len(arr) == 1:
            result.extend([x for x in arr[0] if x != -999])
        else:
            num_elements = len(arr[0])
            sums = np.zeros(num_elements)
            counts = np.zeros(num_elements)
            for sub_arr in arr:
                for i, value in enumerate(sub_arr):
                    if value != -999:
                        sums[i] += value
                        counts[i] += 1

            for i in range(num_elements):
                if counts[i] > 0:
                    result.append(sums[i] / counts[i])
                else:
                    result.append(-999)

    return result

# 做标准化归一化
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = (data - minVals) / ranges
    return normData
