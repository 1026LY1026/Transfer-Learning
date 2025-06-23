from tf_model import TransAm
import torch.nn as nn
import torch
import torch.optim as optim
import train_data
from sklearn.metrics import r2_score
from datetime import datetime
import test_data
from pyplot_make import *
from utility import *
import lstm_model

USE_MULTI_GPU = True
# 设置默认的CUDA设备
torch.cuda.set_device(0)

# 初始化CUDA环境
torch.cuda.init()

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
    for X, y in loader:  # X--[batch,seq,feature_size]  y--[batch,seq]
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
            de=enc_inputs[-pre_len:, :, :][:,:,0].reshape(1,-1,1)

            loss = criterion(output, y)
            epoch_loss += loss.item()

            pres = output.detach().cpu().numpy()  # [seq, batch, out_size]
            tru = y.detach().cpu().numpy()
            de = de.detach().cpu().numpy()

            y_pre.append(pres.transpose(0, 1, 2).reshape(-1, 1))
            y_true.append(tru.transpose(0, 1, 2).reshape(-1, 1))
            y_depth.append(de.transpose(0, 1, 2).reshape(-1, 1))

            true_size.append(y)

    pre = np.concatenate(y_pre, axis=0)
    true = np.concatenate(y_true, axis=0)
    depth=np.concatenate(y_depth, axis=0)

    pre = np.nan_to_num(pre, nan=0.0)
    true = np.nan_to_num(true, nan=0.0)
    depth = np.nan_to_num(depth, nan=0.0)

    acc = r2_score(true, pre)

    return acc, epoch_loss, true, pre, depth




print(MULTI_GPU)
deviceCount = torch.cuda.device_count()
torch.cuda.set_device(device)
print(deviceCount)
# if MULTI_GPU:
#     model = nn.DataParallel(model, device_ids=device_ids)
train_loader, train_feature_size, train_out_size = train_data.dataset()
test_loader, test_feature_size, test_out_size = test_data.dataset()
model = TransAm(train_feature_size, train_out_size).to(device)
#
# model = lstm_model.lstm_uni_attention(input_size=11, hidden_size=256, num_layers=2,
#                                              pre_length=50, seq_length=50).to(device)
# model.load_state_dict(torch.load('./file/Model_F4.pkl', map_location=torch.device('cuda')))
criterion = nn.MSELoss()  # 忽略 占位符 索引为0.9
optimizer = optim.SGD(model.parameters(), lr=tf_lr, momentum=0.99)
# optimizer = optim.Adam(model.parameters(), lr=tf_lr, weight_decay=0.001)
# if MULTI_GPU:
#     optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
print(device)


def initiate():
    train_acc_size = []
    test_acc_size = []
    train_loss_size = []
    test_loss_size = []
    att_size = []
    start = datetime.now()
    # model = train_model.train()
    for epoch in range(5000):
        model.train()
        train(model, train_loader)
        model.eval()
        train_acc, train_loss, true_train, pre_train, train_depth = test(model, train_loader)
        train_acc_size.append(train_acc)
        train_loss_size.append(train_loss)
        print('Epoch:', '%04d' % epoch, 'loss =', '{:.6f}'.format(train_loss), ' acc =',
              '{:.6f}'.format(train_acc), 'time = ', start)

        test_acc, test_loss, true_csv, pre_csv, test_depth = test(model, test_loader)

        test_acc_size.append(test_acc)
        test_loss_size.append(test_loss)
        print('TEST:', 'loss =', '{:.6f}'.format(test_loss), ' acc =',
              '{:.6f}'.format(test_acc), "time = ", datetime.now())
        if epoch % 100 == 0:
            torch.save(model.state_dict(), './result/train_model/Model_F4_9.pkl')

            train_de = pd.DataFrame(np.concatenate(train_depth, axis=0), columns=['train_depth'])
            train_t = pd.DataFrame(np.concatenate(true_train, axis=0), columns=['train_true'])
            train_p = pd.DataFrame(np.concatenate(pre_train, axis=0), columns=['train_pre'])

            test_de = pd.DataFrame(np.concatenate(test_depth, axis=0), columns=['train_depth'])
            test_t = pd.DataFrame(np.concatenate(true_csv, axis=0), columns=['test_true'])
            test_p = pd.DataFrame(np.concatenate(pre_csv, axis=0), columns=['test_pre'])

            csv_t = pd.concat([train_de, train_t, train_p, test_de, test_t, test_p], axis=1)
            csv_t.to_csv('./result/data/rel_pre_S_F4_9.csv', sep=",", index=True)

        loss_acc_dict = {'train_loss': train_loss_size, 'test_loss': test_loss_size,
                         'train_acc': train_acc_size, 'test_acc': test_acc_size}
        loss_acc = pd.DataFrame(loss_acc_dict)
        loss_acc.to_csv('./result/data/loss_acc_S-F4_9.csv', sep=",", index=True)
        if epoch == 10000 :
            plt.figure(figsize=(6,5))
            plt.plot(loss_acc['train_loss'],'-o',markersize=3)
            plt.ylabel("train_loss",fontsize=18)
            plt.xlabel("epoch",fontsize=18)
            plt.grid()
            plt.savefig('./result/png/loss/loss_train_S-F4_9.png')

            plt.figure(figsize=(6, 5))
            plt.plot(loss_acc['test_loss'], '-o', markersize=3)
            plt.ylabel("test_loss", fontsize=18)
            plt.xlabel("epoch", fontsize=18)
            plt.grid()
            plt.savefig('./result/png/loss/loss_test_S-F4_9.png')

            plt.figure(figsize=(6, 5))
            plt.plot(loss_acc['train_acc'], '-o', markersize=3)
            plt.ylabel("train_acc", fontsize=18)
            plt.xlabel("epoch", fontsize=18)
            plt.grid()
            plt.savefig('./result/png/acc/acc_train_S-F4_9.png')

            plt.figure(figsize=(6, 5))
            plt.plot(loss_acc['test_acc'], '-o', markersize=3)
            plt.ylabel("test_acc", fontsize=18)
            plt.xlabel("epoch", fontsize=18)
            plt.grid()
            plt.savefig('./result/png/acc/acc_test_S-F4_9.png')
            plt.show()



initiate()