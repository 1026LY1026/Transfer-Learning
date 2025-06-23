import numpy as np
import pandas as pd

# 数据处理层
# 本地
data_path_4 = './data/UiS_F4.csv'
data_path_5 = './data/UiS_F5.csv'
data_path_7 = './data/UiS_F7.csv'
# 202
# data_path = '/home/RH/anaconda3/envs/nlp/App/student0/tft/data/BZ19-6-1-change.csv'
# data_path_5 = '/home/RH/anaconda3/envs/nlp/App/student0/tft/data/data_05.xls'
# data_path_6 = '/home/RH/anaconda3/envs/nlp/App/student0/tft/data/data_06.xlsx'
# data_path_7 = '/home/RH/anaconda3/envs/nlp/App/student0/tft/data/data_07.xlsx'
# 10
# data_path = '/home/eg840/zfq/BZ19-6-1.csv'
# data_path_4 = '/home/eg840/zfq/tft/fin_tf_used_data/UiS_F4.csv'
# data_path_5 = '/home/eg840/zfq/tft/fin_tf_used_data/UiS_F5.csv'
# data_path_7 = '/home/eg840/zfq/tft/fin_tf_used_data/UiS_F7.csv'
# data_path_9 = '/home/eg840/zfq/tft/fin_tf_used_data/UiS_F9.csv'
# data_path_9A = '/home/eg840/zfq/tft/fin_tf_used_data/UiS_F9A.csv'
# data_path_10 = '/home/eg840/zfq/tft/fin_tf_used_data/UiS_F10.csv'
# data_path_12 = '/home/eg840/zfq/tft/fin_tf_used_data/UiS_F12.csv'
# data_path_14 = '/home/eg840/zfq/tft/fin_tf_used_data/UiS_F14.csv'
# data_path_15A = '/home/eg840/zfq/tft/fin_tf_used_data/UiS_F15A.csv'

# plt path
train_rop_plt_path = './train/png/rop'
train_line_plt_path = './train/png/line'
train_rel_plt_path = './train/png/rel'

test_rop_plt_path = './test/png/rop'
test_line_plt_path = './test/png/line'
test_rel_plt_path = './test/png/rel'

# 模型路径
model_path = './file/model.pkl'

# 选择关键字
key_word = 'GR'

# 选择特征列
# elements_05 = ['FLOWOUT', 'ROP', 'Pump 1', 'Pump 2', 'Pump 3', 'ROP_log m/hr', 'BitTime']
# elements_06 = ['Vertical Depth', 'ROP', 'FlowOut', 'HKH', 'Pump 1', 'Pump 2', 'Pump 3', 'BitTime', 'BitRun', 'DH RPM']
# elements_07 = ['Pump 1', 'Pump 2', 'Pump 3', 'HKH', 'FlowOut', 'MW out', 'DEXPONENT', 'DH RPM',
#                'ROP', 'BitTime', 'BitRun']
elements = ['Unnamed: 0', 'FORMATION', 'Gamma gAPI']

# 最大最小距离，可能用于数据预处理或模型训练的超参数
max_min_distance = 500

# 每个训练批次中的样本数量，即批大小
batch = 1

# 预测序列的长度，可能表示模型要预测多步时间序列
pre_len = 10

# 输入序列的长度，即模型看到的历史信息的长度
seq_len = 20

# 数据采样间隔，可能表示每隔多少时间步采样一个样本
interval = 1

# TensorFlow模型的学习率
tf_lr = 1e-5

# 模型训练过程中的一个指标，可能是准确度
best_acc = 0.99

#
distance_elements = ['1', '2', '3', '4', '5', '6', '7', '8','9', '10', '11', '12', '13', '14', '15', '16',
                     '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
                     '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46',
                     '47', '48', '49', '50']

# 获取最用保存数据路径
def get_test_result_path(epoch):
    path = "./test/data/result" + str(epoch) + ".csv"
    return path
def get_train_result_path(epoch):
    path = "./train/data/result" + str(epoch) + ".csv"
    return path


# 做标准化归一化
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = (data - minVals) / ranges
    return normData


# 相对误差
def mean_relative_error(y_true, y_pred):
    y_true = y_true.tolist()
    y_pred = y_pred.tolist()
    if y_true == 0:
        relative_error = np.array(y_true)
    else:
        relative_error = np.average(np.abs(y_true - y_pred) / y_true)
    return relative_error


# 将数据变为有监督学习
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # 判断数据是单变量还是多变量
    n_vars = 1 if type(data) is list else data.shape[1]
    # 创建DataFrame对象
    df = pd.DataFrame(data)
    # 存储转化后的列和列名
    cols, names = [], []
    # i: n_in, n_in-1, ..., 1
    # 代表t-n_in, ... ,t-1
    # 创建输入序列的列
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(data.columns[j] + '(t-%d)' % (i)) for j in range(n_vars)]
    # 创建输出序列的列
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(data.columns[j] + '%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [(data.columns[j] + '(t+%d)' % (i)) for j in range(n_vars)]
    # 合并所有列
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 删除包含NaN值的行
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# 计算真实值和预测值之间的平均相对误差
def take_rela_error(true, pre):
    # 将真实值和预测值展平成一维数组
    true = true.flatten()
    pre = pre.flatten()
    # 存储每个真实值和预测值对应的平均相对误差
    rel_size = []
    # 遍历真实值和预测值，并计算平均相对误差
    for t, p in zip(true, pre):
        rel_size.append(mean_relative_error(t, p).tolist())
    # 返回平均相对误差的平均值
    return np.mean(rel_size)
