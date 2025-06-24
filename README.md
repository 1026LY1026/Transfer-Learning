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

## tf_rop/train文件夹下代码说明



| 文件名                      | 功能描述          |
| :----------------------- | :------------ |
| fineD\_bz10.py           | 模型冻结实验（BH 油田） |
| fineD\_well3.py          | 模型冻结实验（XJ 油田） |
| fine\_tuning\_well\_3.py | 模型微调实验（XJ 油田） |
| fine\_tunning\_bz10.py   | 模型微调实验（BH 油田） |
| pre\_train.py            | 模型预训练         |
| pyplot\_make.py          | 实验数据画图        |
| rop\_plot.py             | 实验数据画图        |
| test.py                  | 基于训练好的模型进行测试  |
| test\_data.py            | 测试集数据预处理      |
| tf\_model.py             | 模型            |
| train.py                 | 模型训练          |
| train\_data.py           | 训练集数据预处理      |
| train\_test\_plot.py     | 测试结果画图        |
| trm\_model.py            | 模型迁移冻结        |
| utility.py               | 模型参数设置        |



```

