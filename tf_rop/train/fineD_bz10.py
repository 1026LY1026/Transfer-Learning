from trm_model import TransAm
import torch.nn as nn
import torch
import torch.optim as optim
import train_data
import test_data
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from datetime import datetime
from train_test_plot import *
import os
from utility import *

USE_MULTI_GPU = True
# 设置默认的CUDA设备
torch.cuda.set_device(0)

# 初始化CUDA环境
torch.cuda.init()
#
# # 检测机器是否有多张显卡
# if USE_MULTI_GPU and torch.cuda.device_count() > 1:
#     MULTI_GPU = True
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"  # 设置所有六张显卡的编号
#     device_ids = list(range(6))  # 设置所有六张显卡的编号
# else:
#     MULTI_GPU = False
# device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# 检测机器是否有多张显卡
if USE_MULTI_GPU and torch.cuda.device_count() > 1:
    MULTI_GPU = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置所有六张显卡的编号
    device_ids = ['0'] # 设置所有六张显卡的编号
else:
    MULTI_GPU = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(TModel, loader):
    epoch_loss = 0
    for X, y in loader:
     # X--[batch,seq,feature_size]  y--[batch,seq]
        X, y = X.to(device), y.to(device)
        enc_inputs = X.permute([1, 0, 2])  # [seq,batch,feature_size]
        y = y.permute([1, 0, 2])
        key_padding_mask = torch.zeros(enc_inputs.shape[1], enc_inputs.shape[0], dtype=torch.float32)
        # mask_pad = torch.zeros(enc_inputs.shape[1], 50, dtype=torch.float32) # [batch,seq]

        # key_padding_mask[:, seq_len-pre_len:] = mask_pad
        # key_padding_mask = key_padding_mask.bool().masked_fill(key_padding_mask == 0, True).masked_fill(
        #     key_padding_mask == 1, False)
        key_padding_mask = key_padding_mask.to(device)
        # key_padding_mask = None

        optimizer.zero_grad()
        mask = (torch.triu(torch.ones(y.size(0), y.size(0))) == 1).transpose(0, 1)
        tgt_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)

        output = TModel(enc_inputs, key_padding_mask, y,tgt_mask)
        # output = TModel(enc_inputs, key_padding_mask, y)
        output = output[-pre_len:, :, :]
        y = y[-pre_len:, :, :]
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(TModel.parameters(), 0.10)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss


# 测试函数
def test(TModel, tf_loader):
    epoch_loss = 0
    y_pre = []
    y_true = []
    true_size = []
    y_depth = []

    for x, y in tf_loader:
        with torch.no_grad():
            x, y = x.to(device), y.to(device)

            enc_inputs = x.permute([1, 0, 2])  # [seq, batch, feature_size]
            y = y.permute([1, 0, 2])
            key_padding_mask = torch.zeros(enc_inputs.shape[1], enc_inputs.shape[0], dtype=torch.float32).to(device)

            attn_mask = None
            mask = (torch.triu(torch.ones(y.size(0), y.size(0))) == 1).transpose(0, 1)
            tgt_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to(device)
            output = TModel(enc_inputs, key_padding_mask, y,tgt_mask)
            # output = TModel(enc_inputs, key_padding_mask, y)

            output = output[-pre_len:, :, :]
            y = y[-pre_len:, :, :]
            de=enc_inputs[-pre_len:, :, :][:,:,0].unsqueeze(-1)

            loss = criterion(output, y)
            epoch_loss += loss.item()

            pres = output.detach().cpu().numpy()  # [seq, batch, out_size]
            pres=pres.transpose(0, 1, 2)
            pres = np.squeeze(pres, axis=2)
            pres_ = extract_anti_diagonal_blocks(pres)
            pres_ = np.array(pres_)
            pres_ = pres_[:, np.newaxis]

            tru = y.detach().cpu().numpy()
            tru=tru.transpose(0, 1, 2)
            tru = np.squeeze(tru, axis=2)
            tru_ = extract_anti_diagonal_blocks(tru)
            tru_ = np.array(tru_)
            tru_ = tru_[:, np.newaxis]

            de = de.detach().cpu().numpy()
            de=de.transpose(0, 1, 2)
            de = np.squeeze(de, axis=2)
            de_ = extract_anti_diagonal_blocks(de)
            de_ = np.array(de_)
            de_ = de_[:, np.newaxis]


            y_pre.append(pres_)
            y_true.append(tru_)
            y_depth.append(de_)

            true_size.append(y)

    pre = np.concatenate(y_pre, axis=0)
    true = np.concatenate(y_true, axis=0)
    depth=np.concatenate(y_depth, axis=0)

    pre = np.nan_to_num(pre, nan=0.0)
    true = np.nan_to_num(true, nan=0.0)
    depth = np.nan_to_num(depth, nan=0.0)

    acc = r2_score(true, pre)

    mse = mean_squared_error(true, pre)

    mae = mean_absolute_error(true, pre)

    return acc,mse,mae, epoch_loss, true, pre, depth



print(MULTI_GPU)
deviceCount = torch.cuda.device_count()
torch.cuda.set_device(device)
print(deviceCount)
# if MULTI_GPU:
#     model = nn.DataParallel(model, device_ids=device_ids)
#train_loader, train_feature_size, train_out_size = test_data.dataset(seq_len,data_gra_2)
# train_loader, train_feature_size, train_out_size = train_data.dataset(seq_len)
train_loader, train_feature_size, train_out_size = train_data.dataset(seq_len)
test_loader, test_feature_size, test_out_size = test_data.dataset(seq_len,data_path_bz_6)
model=TransAm(train_feature_size, train_out_size).to(device)
# model = TransAm(train_feature_size, train_out_size).to(device)

model.load_state_dict(torch.load('tf/train/out_model/volve/Model_volve_10.pkl', map_location=torch.device('cuda')))
criterion = nn.MSELoss()  # 忽略 占位符 索引为0.9
optimizer = optim.SGD(model.parameters(), lr=tf_lr, momentum=0.99)
#optimizer = optim.Adam(model.parameters(), lr=tf_lr, weight_decay=0.001)
# if MULTI_GPU:
#     optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
print(device)
 
def initiate():
    train_acc_size = []
    train_mse_size=[]
    train_mae_size=[]
    train_loss_size = []

    test_acc_size = []
    test_mse_size=[]
    test_mae_size=[]
    test_loss_size = []

    start = datetime.now()
    for epoch in range(2500):
        model.train()
        train(model, train_loader)
        model.eval()
        train_acc, train_mse,train_mae,train_loss, true_train, pre_train, train_depth = test(model, train_loader)
        train_mse_size.append(train_mse)
        train_mae_size.append(train_mae)
        train_acc_size.append(train_acc)
        train_loss_size.append(train_loss)
        print('Epoch:', 'train %04d' % epoch, 'loss =', '{:.6f}'.format(train_loss), ' acc =','{:.6f}'.format(train_acc),
        ' mse =', '{:.6f}'.format(train_mse),' mae =', '{:.6f}'.format(train_mae),'time = ', start)

        test_acc, test_mse, test_mae, test_loss, true_test, pre_test, test_depth = test(model, test_loader)
        test_mse_size.append(test_mse)
        test_mae_size.append(test_mae)
        test_acc_size.append(test_acc)
        test_loss_size.append(test_loss)
        print('Epoch:', 'test %04d' % epoch, 'loss =', '{:.6f}'.format(test_loss), ' acc =', '{:.6f}'.format(test_acc),
              ' mse =', '{:.6f}'.format(test_mse), ' mae =', '{:.6f}'.format(test_mae), 'time = ', start)

        loss_acc_mse_mae_dict = {'train_loss': train_loss_size, 'test_loss': test_loss_size,
                                 'train_acc': train_acc_size, 'test_acc': test_acc_size,
                                 'train_mse': train_mse_size, 'train_mae': train_mae_size,
                                 'test_mse': test_mse_size, 'test_mae': test_mae_size, }
        loss_acc_mse_mae = pd.DataFrame(loss_acc_mse_mae_dict)
        loss_acc_mse_mae.to_csv('tf/train/D/bz_de/loss_acc_mse_mae6.csv', sep=",", index=True)

        if epoch % 10 == 0:
            if epoch >=0:
                torch.save(model.state_dict(), 'tf/train/out_model/Model_bz_de6.pkl')

                train_de = pd.DataFrame(np.concatenate(train_depth, axis=0), columns=['train_depth'])
                train_t = pd.DataFrame(np.concatenate(true_train, axis=0), columns=['train_true'])
                train_p = pd.DataFrame(np.concatenate(pre_train, axis=0), columns=['train_pre'])

                test_de = pd.DataFrame(np.concatenate(test_depth, axis=0), columns=['test_depth'])
                test_t = pd.DataFrame(np.concatenate(true_test, axis=0), columns=['test_true'])
                test_p = pd.DataFrame(np.concatenate(pre_test, axis=0), columns=['test_pre'])

                # csv_train = pd.concat([train_de,train_t,train_p], axis=1)
                # csv_train.to_csv('tf/train/D/bz_en/rel_pre_train_1.csv', sep=",", index=True)

                # csv_test = pd.concat([test_de, test_t, test_p], axis=1)
                # csv_test.to_csv('tf/train/D/bz_en/rel_pre_test_1.csv', sep=",", index=True)

                acc_loss_plot(loss_acc_mse_mae['train_loss'], loss_acc_mse_mae['test_loss'], 'loss',
                              'tf/train/D/bz_de/loss_6.png')
                acc_loss_plot(loss_acc_mse_mae['train_acc'], loss_acc_mse_mae['test_acc'], 'acc',
                              'tf/train/D/bz_de/acc_6.png')

                # true_test_plot(csv_train['train_depth'], csv_train['train_true'], csv_train['train_pre'], 'train',
                #                'tf/train/gramma/png/pre_true_train_2_7.png')
                # true_test_plot(csv_test['test_depth'], csv_test['test_true'], csv_test['test_pre'], 'test',
                #                'tf/train/output/bz/300_50_20/pre_true_test_volve_15.png')


initiate()