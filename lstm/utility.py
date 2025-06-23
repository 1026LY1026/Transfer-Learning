import numpy as np
import pandas as pd

data_path_4 = './data/volve/S_F4.csv'
data_path_5 = './data/volve/S_F5.csv'
data_path_10 = './data/volve/S_F10.csv'
data_path_12 = './data/volve/S_F12.csv'
data_path_14 = './data/volve/S_F14.csv'

data_path_well_2 = './data/xj/well_2.csv'
data_path_well_3 = './data/xj/well_3.csv'


data_path_bz_6='./data/bz/BZ19_6_6.csv'
data_path_bz_10='./data/bz/BZ19_6_10.csv'



model_pre_len = 50
model_seq_len = 300
model_tf_lr = 0.00005 # 0.0005
model_batch = 128
model_feature_size=12
model_d_model=512
model_num_layers=1
model_dropout=0.01

def averages(matrix):  # 计算平均值
    matrix = np.array(matrix)
    row_count, col_count = matrix.shape
    max_diagonal = row_count + col_count - 1
    diagonals = np.zeros(max_diagonal)
    counts = np.zeros(max_diagonal, dtype=int)
    for i in range(row_count):
        for j in range(col_count):
            num = matrix[i, j]
            diagonal_index = i + j
            diagonals[diagonal_index] += num
            counts[diagonal_index] += 1
    averages = diagonals / counts
    return averages