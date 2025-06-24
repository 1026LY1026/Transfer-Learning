# Improving Generalization in Rate of Penetration Prediction through Transfer Learning across Oil Fields
Improving Generalization in Rate of Penetration Prediction through Transfer Learning across Oil Fields：以下是论文中迁移学习的原始代码实现
# 安装和使用说明
## 环境
**python 3.6.10
cuda:11**

| 包名         | 版本号   |
|:---------------|:---------|
| DateTime       | 4.9      |
| Pillow         | 8.4.0    |
| PyWavelets     | 1.1.1    |
| asgiref        | 3.3.4    |
| certifi        | 2021.5.30|
| cycler         | 0.11.0   |
| dataclasses    | 0.8      |
| decorator      | 4.4.2    |
| et-xmlfile     | 1.1.0    |
| imageio        | 2.15.0   |
| joblib         | 1.1.1    |
| kiwisolver     | 1.3.1    |
| matplotlib     | 3.3.4    |
| networkx       | 2.5.1    |
| numpy          | 1.19.5   |
| openpyxl       | 3.0.9    |
| pandas         | 1.1.5    |
| pip            | 21.3.1   |
| psycopg2       | 2.8.6    |
| pyparsing      | 3.1.4    |
| python-dateutil| 2.9.0.post0 |
| pytz           | 2024.2   |
| scikit-image   | 0.17.2   |
| scikit-learn   | 0.24.2   |
| scipy          | 1.5.4    |
| setuptools     | 59.6.0   |
| six            | 1.16.0   |
| sqlparse       | 0.4.1    |
| threadpoolctl  | 3.1.0    |
| tifffile       | 2020.9.3 |
| torch          | 1.10.1   |
| torchaudio     | 0.10.1   |
| torchvision    | 0.11.2   |
| typing-extensions | 4.1.1 |
| wheel          | 0.37.1   |
| zope.interface | 5.5.2    |
## 参数说明

| 参数名          | 含义      |
| :----------- | :------ |
| train\_batch | 训练集批次   |
| test\_batch  | 测试集批次   |
| pre\_len     | 输出序列长度  |
| seq\_len     | 输入序列长度  |
| interval     | 数据集读取间隔 |
| tf\_lr       | 学习率     |




## 代码说明

GRU：GRU对比试验部分代码
lstm：LSTM对比实验部分代码
tf_rop/train：迁移学习实验部分代码
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7b4bf8d0a5bd4d218d78dccf204e5e74.png)
## tf_rop/train文件夹下代码说明


fineD_bz10.py   模型冻结实实验 （BH油田）

fineD_well3.py  模型冻结实验 （XJ油田）

fine_tuning_well_3.py

fine_tunning_bz10.py

pre_train.py

pyplot_make.py

rop_plot.py

setup.py

test.py

test_data.py

tf_model.py

train.py

train_data.py

train_test_plot.py

trm_model.py

utility.py


```

