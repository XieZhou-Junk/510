import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


def get_data(file_name):
    data = pd.read_csv(file_name)
    lowest_list = []
    for i in data['lowest']:
        lowest_list.append(float(i))
    return np.array(lowest_list[::-1])


lowest = get_data('Price Data.csv')  # 获取最低价数组
size = len(lowest)
time = np.arange(size)
time_predict = np.arange(size + 1)  # 用于预测的时间轴 多一个值
degree = [7, 8, 9, 10]  # 多次尝试后 选用7~10次多项式函数拟合
predict_list = []
plt.figure(figsize=(19.20, 10.80))
for d in degree:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('linear', linear_model.Ridge())])  # 创建多项式且具有l2正则化的线性最小二乘法模型
    model.fit(time[:, np.newaxis], lowest)  # 进行拟合
    lowest_fit_predict = model.predict(time_predict[:, np.newaxis])  # 输出包含预测的拟合函数结果
    plt.plot(time_predict, lowest_fit_predict, linewidth=2)  # 画线
    print(d, int(lowest_fit_predict[-1]))
    predict_list.append(lowest_fit_predict[-1])  # 收集4个拟合函数的预测值
predict = np.int(np.average(predict_list))  # 求得预测值平均数
print('avg', predict)
plt.scatter(size, predict, s=100)
plt.plot(lowest, linewidth=5)
plt.grid()
plt.show()
