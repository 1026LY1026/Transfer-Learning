import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from utility import *


def single_data():
    # data_4 = pd.read_csv(data_path_4)
    # data_5 = pd.read_csv(data_path_5) 
    # data_10 = pd.read_csv(data_path_10)
    # data_14 = pd.read_csv(data_path_14)
    # data_all = pd.concat([data_10,data_4,data_14,data_5])
    bz_data_1 = pd.read_csv(data_path_bz_1)
    bz_data_15 = pd.read_csv(data_path_bz_15)
    bz_data_10 = pd.read_csv(data_path_bz_10)
    data_all = pd.concat([bz_data_15,bz_data_1, bz_data_10])
    #bz_data_10=bz_data_10[['MD','RPMA','ROPA']]
    
  #  well_3=pd.read_csv(data_path_well_3)
  #  data_all = bz_data_10
    
    data_all = data_all.astype('float32')
    x_hat = data_all.values
    zero_size = []

    # x_T 中第二个元素（可能是某种序列数据）中值为0的位置索引，并将这些索引值保存到 zero_size 列表中。
    x_T = x_hat.T
    for index, elem in enumerate(x_T):  # 同时返回索引和元素值
        if index == 1:
            for i, e in enumerate(elem):
                if e == 0:
                    zero_size.append(i)
        minVals = elem.min(0)
        maxVals = elem.max(0)
        # 当列的数据差距过大做归一化
        if maxVals-minVals > 10000:
            elem = noramlization(elem)
            x_T[index] = elem

    # 做倒置
    # x = np.flipud(x_hat)
    x = x_hat
    y = x[:,-1]

    if len(y.shape) < 2:
        y = np.expand_dims(y, 1)
    x = np.nan_to_num(x, nan=0.0)
    return x, y  # [none,feature_size]  [none,feature_size]默认out_size为1


def data_load(seq_len):
    x, y = single_data()
    len = x.shape[0]
    data_last_index = len - seq_len
    X = []
    Y = []
    for i in range(0, data_last_index, 10):
        data_x = np.expand_dims(x[i:i + seq_len], 0)  # [1,seq,feature_size]
        data_y = np.expand_dims(y[i:i + seq_len], 0)  # [1,seq,out_size]
        # data_y=np.expand_dims(y[,0)   #[1,seq,out_size]
        X.append(data_x)
        Y.append(data_y)

    # del X[-interval:]
    # del Y[0:interval]
    data_x = np.concatenate(X, axis=0)
    data_y = np.concatenate(Y, axis=0)
    data = torch.from_numpy(data_x).type(torch.float32)
    label = torch.from_numpy(data_y).type(torch.float32)
    return data, label  # [num_data,seq,feature_size]  [num_data,seq] 默认out_size为1


def dataset(seq_len):
    X, Y = data_load(seq_len)
    feature_size = X.shape[-1]
    out_size = Y.shape[-1]

    dataset_train = TensorDataset(X, Y)
    dataloader = DataLoader(dataset_train, batch_size=train_batch, shuffle=False)
    return dataloader, feature_size, out_size


