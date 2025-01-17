import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import curve_fit
from scipy.stats import linregress  # 导入线性回归工具
import os  # 导入os模块，用于获取文件名
import matplotlib.ticker as ticker

# 设置主刻度间隔，使用 MultipleLocator 控制
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# 定义第二类修正贝塞尔函数模型
def bessel_model(x, a, b, c):
    return a * special.kv(0, b * x) + c  # 第二类修正贝塞尔函数 (Kv)

# 文件路径列表
file_paths = [
    r'E:\清华云盘\陈显力_1\我的资料库\调研\碳纳米管薄膜气体温度场\实验数据\20241227\修正\1cm\数据提取\xlsx文件\更改名字\35V-200sccm.xlsx',
    r'E:\清华云盘\陈显力_1\我的资料库\调研\碳纳米管薄膜气体温度场\实验数据\20241227\修正\1cm\数据提取\xlsx文件\更改名字\35V-400sccm.xlsx',
    r'E:\清华云盘\陈显力_1\我的资料库\调研\碳纳米管薄膜气体温度场\实验数据\20241227\修正\1cm\数据提取\xlsx文件\更改名字\35V-600sccm.xlsx'
]

# 设置图表
plt.figure(figsize=(12, 9))

# 固定区间，例如选择 x 范围 [0.5, 2.0]
x_min = 1
x_max = 3

# 遍历所有文件
for file_path in file_paths:
    # 读取Excel文件中的数据
    df = pd.read_excel(file_path)

    # 提取x轴和y轴数据
    x_data = df.iloc[:, 0]  # 第一列为x轴
    y_data = df.iloc[:, 1]  # 第二列为y轴

    # 获取文件名，使用os.path.basename来处理路径
    file_name = os.path.basename(file_path)

    # 过滤出指定区间内的x、y数据
    mask = (x_data >= x_min) & (x_data <= x_max)
    x_filtered = x_data[mask]
    y_filtered = y_data[mask]

    # 对指定区间的数据进行线性拟合
    slope, intercept, r_value, p_value, std_err = linregress(x_filtered, y_filtered)

    # 打印斜率
    print(f'For file {file_name}, the slope in the range x = [{x_min}, {x_max}] is: {slope:.4f}(K/mm)')

    # 使用curve_fit进行贝塞尔函数拟合
    params, covariance = curve_fit(bessel_model, x_data, y_data, p0=[1, 1, 1],maxfev=10000)  # p0为初始参数猜测的初始值

    # 提取拟合参数
    a_fit, b_fit, c_fit = params

    # 计算拟合的y值
    y_fit = bessel_model(x_data, *params)

    # 计算残差和R²
    residuals = y_data - y_fit
    ss_res = np.sum(residuals**2)  # 残差平方和
    ss_tot = np.sum((y_data - np.mean(y_data))**2)  # 总平方和
    r_squared = 1 - (ss_res / ss_tot)  # 计算R²

    # 打印拟合参数和拟合程度
    print(f'For file {file_name}, the fitted parameters are:')
    print(f'a = {a_fit:.4f}, b = {b_fit:.4f}, c = {c_fit:.4f}')
    print(f'R² = {r_squared:.4f}')
    print(f'Fitted equation: y = {a_fit:.4f} * Kv(0, {b_fit:.4f} * x) + {c_fit:.4f}\n')

    # 创建细分的x值进行拟合曲线
    x_fit = np.linspace(min(x_data), max(x_data), 500)
    y_fit_curve = bessel_model(x_fit, *params)

    # 移除文件扩展名
    file_name_no_ext = file_name.rsplit('.', 1)[0]

    # 绘制散点图
    plt.scatter(x_data, y_data, label=f'{file_name_no_ext} - Data Points') # 使用文件名来区分不同数据集

    # 绘制拟合曲线
    plt.plot(x_fit, y_fit_curve, label=f'{file_name_no_ext} - Fitted Curve')

# 设置刻度字体大小
plt.tick_params(axis='both', which='major', labelsize=18)

# 设置主刻度和次刻度
ax = plt.gca()  # 获取当前坐标轴
ax.xaxis.set_major_locator(MultipleLocator(2))  # 设置x轴的主刻度间隔为2
ax.xaxis.set_minor_locator(AutoMinorLocator(4))  # 设置x轴的次刻度数量为4

ax.yaxis.set_major_locator(MultipleLocator(100))  # 设置y轴的主刻度间隔为100
ax.yaxis.set_minor_locator(AutoMinorLocator(5))  # 设置y轴的次刻度数量为5

# 绘制局部线性拟合区间
plt.axvspan(x_min, x_max, color='yellow', alpha=0.3, label=f'Linear Fit Range')  # 绘制拟合区间

# 添加区间说明文本
# 在 x_min + x_max 的中点基础上加 0.2，往右移动
x_text_pos = (x_min + x_max) / 2 + 1

plt.text(x_text_pos,
         max(y_data),
         f'Linear Fit Range: x = [{x_min}, {x_max}]',
         horizontalalignment='center',
         fontsize=14,
         color='black',
         weight='bold')

# plt.text((x_min + x_max) / 2, max(y_data), f'Linear Fit Range: x = [{x_min}, {x_max}]',
#              horizontalalignment='center', fontsize=14, color='black', weight='bold')

# 添加标签和图例，并调整标签字体大小
plt.xlabel('Location (mm)', fontsize=20)  # 设置 x 轴标签字体大小为 16
plt.ylabel('Temperature (°C)', fontsize=20)  # 设置 y 轴标签字体大小为 16
plt.legend(fontsize=16)  # 设置图例的字体大小为 14

# 设置横坐标和纵坐标范围
plt.xlim(0, 14.5)
plt.ylim(30, 450)

# 显示图表
plt.show()




