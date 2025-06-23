import warnings
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from pylab import mpl
from utility import *
warnings.filterwarnings('ignore')

# 设置matplotlib的配置
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

def train_data_load():

   # bz_data_10 = pd.read_csv(data_path_bz_10)
    well_3 = pd.read_csv(data_path_well_3)
    data_4 = pd.read_csv(data_path_4)
    data_5 = pd.read_csv(data_path_5)
    data_10 = pd.read_csv(data_path_10)
    data_14 = pd.read_csv(data_path_14)
  #  data_all = pd.concat([data_10,data_4,data_14,data_5])

   # data_all = bz_data_10
    data_all = well_3
    data = data_all.astype('float32')
    data.dropna(inplace=True)
    data = data.values

    data_ =torch.tensor(data[:len(data)])
    maxc, _ = data_.max(dim=0)
    minc, _ = data_.min(dim=0)
    y_max = maxc[-1]
    y_min = minc[-1]
    de_max = maxc[0]
    de_min = minc[0]
    data_ = (data_ - minc) / (maxc - minc)

    data_last_index = data_.shape[0] - model_seq_len

    data_X = []
    data_Y = []
    for i in range(0, data_last_index - model_pre_len+1):
        data_x = np.expand_dims(data_[i:i + model_seq_len], 0)  # [1,seq,feature_size]
        data_y = np.expand_dims(data_[i:i + model_seq_len], 0)  # [1,seq,out_size]
        data_X.append(data_x)
        data_Y.append(data_y)

    data_X=np.concatenate(data_X, axis=0)
    data_Y = np.concatenate(data_Y, axis=0)

    process_data = torch.from_numpy(data_X).type(torch.float32)
    process_label = torch.from_numpy(data_Y).type(torch.float32)

    data_feature_size = process_data.shape[-1]

    dataset_train = TensorDataset(process_data, process_label)

    data_dataloader = DataLoader(dataset_train, batch_size=model_batch, shuffle=False)
    return data_dataloader,y_max,y_min, de_max,de_min


