import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
import torch 
from torch.autograd import Variable
import scipy.io as scio
import csv
import random
import time
import statsmodels.api as sm

# 导入数据
file_dir = 'E:/Research/Data/WSO/gather_harmonic_coefficient.mat'
data_str = scio.loadmat(file_dir)
data_mat = data_str['save_var'] # size: 619,100
hc_lm = np.zeros((617,1))
hc_t = range(0,617)
lm = 10
for i in range(0,617):
    hc_lm[i] = data_mat[i+2][lm]
hc_lm = hc_lm.flatten()

# 创建一个示例序列
np.random.seed(0)
time_series = np.sin(np.linspace(0, 10*np.pi, 200)) + np.random.randn(200)
time_series = hc_lm

# 进行时间序列分解
stl_result = sm.tsa.seasonal_decompose(time_series, model='additive', period=50)
trend = stl_result.trend
seasonal = stl_result.seasonal
residual = stl_result.resid

# 分解结果绘图
plt.figure()

plt.subplot(411)
plt.plot(time_series, label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residual')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# LSTM预测
data0 = residual
data0_full = data0
data0_t_full = np.array(range(0,len(data0_full)))
# data0 = data0[:-120]

data0_t = np.array(range(0,len(data0)))
data = pd.DataFrame(data0)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# 定义look_back
look_back = 150

# 生成训练集和测试集
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i+look_back, :])
    return np.array(dataX), np.array(dataY)
dataX, dataY = create_dataset(data,look_back)

# 划分训练集、验证集、测试集
data_t = np.array(range(0,len(dataX)))
train_size = int(len(data_t) * 0.6)
val_size = int(len(data_t) * 0.2)
test_size = len(data_t) - train_size - val_size

# train_size = int(len(data) * 0.7)
# val_size = int(len(data) * 0.2)
# test_size = len(data) - train_size - val_size
# indices = np.random.permutation(len(data))
# train_data, val_data, test_data = data[indices[:train_size], :], \
#     data[indices[train_size:train_size+val_size], :], data[indices[train_size+val_size:], :]

train_t, val_t, test_t = data_t[:train_size], \
    data_t[train_size:train_size+val_size], data_t[train_size+val_size:]
trainX, valX, testX = dataX[:train_size], \
    dataX[train_size:train_size+val_size], dataX[train_size+val_size:]
trainY, valY, testY = dataY[:train_size], \
    dataY[train_size:train_size+val_size], dataY[train_size+val_size:]

trainX, trainY = trainX.reshape(-1,1,look_back), trainY.reshape(-1,1,1)
valX, valY = valX.reshape(-1,1,look_back), valY.reshape(-1,1,1)
testX, testY = testX.reshape(-1,1,look_back), testY.reshape(-1,1,1)

# 转换为PyTorch张量
trainX = torch.from_numpy(trainX).type(torch.Tensor)
trainY = torch.from_numpy(trainY).type(torch.Tensor)
valX = torch.from_numpy(valX).type(torch.Tensor)
valY = torch.from_numpy(valY).type(torch.Tensor)
testX = torch.from_numpy(testX).type(torch.Tensor)
testY = torch.from_numpy(testY).type(torch.Tensor)

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x, _ = self.lstm(x)
        s,b,h = x.shape
        x = x.reshape(s*b, h)
        x = self.fc(x)
        x = x.view(s,b,-1) 
        return x
    
# 定义模型参数
input_dim = look_back
hidden_dim = look_back*2
num_layers = 3
output_dim = 1
lr = 0.01
num_epochs = 2000

# 定义模型和损失函数
model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 定义早停策略
patience = 10
val_loss_min = np.Inf
val_loss_thres = 0.007
counter = 0

# 训练模型
start_time = time.time()
for epoch in range(num_epochs):
    outputs = model(trainX)
    optimizer.zero_grad()
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    val_outputs = model(valX)
    val_loss = criterion(val_outputs, valY)
    val_loss_np = val_loss.detach().numpy()
    
    # 训练集损失达到阈值以下，跳出循环
    if loss.item() < 1e-4:
        print('Epoch [{}/{}], Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch+1, num_epochs, loss.item(), val_loss_np))
        print("The loss value is reached")
        break
    elif (epoch+1) % 50 == 0:
        print('Epoch [{}/{}], Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch+1, num_epochs, loss.item(), val_loss_np))
    
    # 使用验证集进行早停策略
    if val_loss < val_loss_min:
        counter = 0
        val_loss_min = val_loss
        torch.save(model.state_dict(), 'best_model.pkl')
    else:
        counter += 1
        if (counter >= patience) & (val_loss_np < val_loss_thres) & (loss.item() < 0.003):
            print('Early stopping at Epoch [{}/{}], Train Loss: {:.5f}, Val Loss={:.5f}'.format(epoch+1, num_epochs, loss.item(), val_loss_np))
            break
print('Time Cost(s):', time.time()-start_time)
        
# 加载最佳模型
model.load_state_dict(torch.load('best_model.pkl'))

# 测试模型
model.eval()
test_predict = model(testX)

# 反归一化
train_predict = scaler.inverse_transform([[inner] for inner in outputs.detach().numpy().flatten()])
trainY = scaler.inverse_transform([[inner] for inner in trainY.detach().numpy().flatten()])
val_predict = scaler.inverse_transform([[inner] for inner in val_outputs.detach().numpy().flatten()])
valY = scaler.inverse_transform([[inner] for inner in valY.detach().numpy().flatten()])
test_predict = scaler.inverse_transform([[inner] for inner in test_predict.detach().numpy().flatten()])
testY = scaler.inverse_transform([[inner] for inner in testY.detach().numpy().flatten()])

# 计算RMSE
print('Train RMSE: %.3f' % np.sqrt(mean_squared_error(trainY, train_predict)))
print('Val RMSE: %.3f' % np.sqrt(mean_squared_error(valY, val_predict)))
print('Test RMSE: %.3f' % np.sqrt(mean_squared_error(testY, test_predict)))

# 生成预测未来的输入数据集
def create_future_inputs(dataset,look_back):
    dataX_ext = []
    for i in range(len(dataset)-look_back+1):
        a = dataset[i:(i+look_back)]
        dataX_ext.append(a)
    return np.array(dataX_ext)

# 预测未来的时间序列
future_step = 120
future_t = np.array(range(0,future_step))
future_predict = np.zeros((future_step,1))
data_ext = data
for i in range(future_step):
    dataX_ext = create_future_inputs(data_ext, look_back)
    dataX_ext = dataX_ext.reshape(-1,1,look_back)
    dataX_ext = torch.from_numpy(dataX_ext).type(torch.Tensor)
    outputs_ext = model(dataX_ext)
    future_predict[i] = outputs_ext[-1].detach().numpy()
    data_ext = np.append(data_ext,future_predict[i])
future_predict = scaler.inverse_transform(future_predict)

# 绘制训练集、验证集和测试集的真实值和预测值
plt.figure()
xmin, xmax = min(data0_t), max(future_t + len(data0_t))
plt.hlines(0,xmin,xmax,color='pink',linestyles='dashed')
ymin, ymax = min(data0), max(data0)
plt.vlines(617,ymin,ymax,color='pink',linestyles='dashed')

plt.plot(data0_t_full,data0_full,label='real',color='k')
plt.plot(train_t+look_back,train_predict[:train_size], label='train predict')
plt.plot(val_t+look_back,val_predict, label='val predict')
plt.plot(test_t+look_back,test_predict, label='test predict')
plt.plot(future_t+len(data0_t),future_predict, label='future predict')
plt.legend()
plt.suptitle('g_{}^{}'.format(int(l_cor), int(m_cor)))
plt.show()