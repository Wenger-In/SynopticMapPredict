import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# 生成数据
data = []
for i in range(1, 618):
    if i % 150 == 0:
        data.append(5)
    else:
        data.append(2)

# 转换为时间序列
dates = pd.date_range(start='2023-01-01', periods=len(data), freq='27.3D')
ts = pd.Series(data=data, index=dates)

# 交叉验证选择SARIMA模型参数
tscv = TimeSeriesSplit(n_splits=5)
params = []
for train_index, test_index in tscv.split(ts):
    train_data = ts[train_index]
    test_data = ts[test_index]
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                for P in range(3):
                    for D in range(2):
                        for Q in range(3):
                            try:
                                model = SARIMAX(train_data, order=(p, d, q),
                                                seasonal_order=(P, D, Q, 150 // 27.3))
                                result = model.fit()
                                aic = result.aic
                                if aic < best_aic:
                                    best_aic = aic
                                    best_order = (p, d, q)
                                    best_seasonal_order = (P, D, Q, 150 // 27.3)
                            except:
                                continue
    params.append((best_order, best_seasonal_order))

# 计算平均参数并训练模型
best_order = tuple(np.mean([p[0] for p in params], axis=0).round().astype(int))
best_seasonal_order = tuple(np.mean([p[1] for p in params], axis=0).round().astype(int))
model = SARIMAX(ts, order=best_order, seasonal_order=best_seasonal_order)
result = model.fit()

# 进行30天的预测
forecast = result.forecast(steps=150)

# 绘制原始序列和预测序列的折线图
plt.plot(ts.index, ts.values, label='original')
plt.plot(forecast.index, forecast.values, label='forecast')

# 设置图例和标题
plt.legend()
plt.title('SARIMA forecast with 27.3-day interval')

# 显示图形
plt.show()

