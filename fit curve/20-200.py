import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import curve_fit

# 读取Excel文件中的数据
# 假设Excel文件名为data.xlsx，第一列是x轴数据，第二列是y轴数据
file_path = r'C:\Users\za\Desktop\空间分辨率数据整理\20-200.xlsx'
df = pd.read_excel(file_path)

# 提取x轴和y轴数据
x_data = df.iloc[:, 0]  # 第一列为x轴
y_data = df.iloc[:, 1]  # 第二列为y轴

# 绘制散点图
plt.scatter(x_data, y_data, color='blue', label='Data Points')

# 定义第二类修正贝塞尔函数模型
def bessel_model(x, a, b, c):
    return a * special.kv(0, b * x) + c  # 第二类修正贝塞尔函数 (Kv)

# 使用curve_fit进行贝塞尔函数拟合
params, covariance = curve_fit(bessel_model, x_data, y_data, p0=[3, 1.5, 1]) #p0为初始参数猜测的初始值

# 获取拟合参数
a_fit, b_fit, c_fit = params
print(f"拟合参数: a = {a_fit}, b = {b_fit}, c = {c_fit}")

# 创建细分的x值进行拟合曲线
x_fit = np.linspace(min(x_data), max(x_data), 500)
y_fit = bessel_model(x_fit, *params)

# 绘制拟合曲线
plt.plot(x_fit, y_fit, color='red', label='Fitted Bessel Curve')

# 添加标签和图例
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# 显示图表
plt.show()