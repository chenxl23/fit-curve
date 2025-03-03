import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import curve_fit
from scipy.stats import linregress
import os
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# 定义第二类修正贝塞尔函数模型
def bessel_model(x, a, b, c):
    return a * special.kv(0, b * x) + c  # 第二类修正贝塞尔函数 (Kv)

# 文件路径列表
file_paths = [
    r'E:\清华云盘\陈显力_1\我的资料库\调研\碳纳米管薄膜气体温度场\实验数据\20241227\修正\1cm\数据提取\xlsx文件\更改名字\600sccm\15V.xlsx',
    r'E:\清华云盘\陈显力_1\我的资料库\调研\碳纳米管薄膜气体温度场\实验数据\20241227\修正\1cm\数据提取\xlsx文件\更改名字\600sccm\25V.xlsx',
    r'E:\清华云盘\陈显力_1\我的资料库\调研\碳纳米管薄膜气体温度场\实验数据\20241227\修正\1cm\数据提取\xlsx文件\更改名字\600sccm\35V.xlsx'
]

# 设置图表
plt.figure(figsize=(12, 9))

# 固定区间
x_min = 1
x_max = 3

# 颜色列表
colors = ['blue', 'red', 'green', 'orange']

# 存储自定义图例
custom_legend = []
labels = []

# 遍历所有文件
for i, file_path in enumerate(file_paths):
    # 读取Excel数据
    df = pd.read_excel(file_path)

    # 提取x轴和y轴数据
    x_data = df.iloc[:, 0]
    y_data = df.iloc[:, 1]

    # 获取文件名（去掉扩展名）
    file_name_no_ext = os.path.basename(file_path).rsplit('.', 1)[0]

    # 过滤出指定区间数据
    mask = (x_data >= x_min) & (x_data <= x_max)
    x_filtered = x_data[mask]
    y_filtered = y_data[mask]

    # 线性拟合
    slope, intercept, r_value, p_value, std_err = linregress(x_filtered, y_filtered)
    print(f'For file {file_name_no_ext}, slope in range x=[{x_min}, {x_max}]: {slope:.4f} K/mm')

    # 贝塞尔函数拟合
    params, _ = curve_fit(bessel_model, x_data, y_data, p0=[1, 1, 1], maxfev=10000)
    a_fit, b_fit, c_fit = params

    # 计算拟合y值
    y_fit = bessel_model(x_data, *params)

    # 计算 R²
    residuals = y_data - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # 打印拟合结果
    print(f'For file {file_name_no_ext}, fitted parameters:')
    print(f'a = {a_fit:.4f}, b = {b_fit:.4f}, c = {c_fit:.4f}')
    print(f'R² = {r_squared:.4f}')
    print(f'Fitted equation: y = {a_fit:.4f} * Kv(0, {b_fit:.4f} * x) + {c_fit:.4f}\n')

    # 生成平滑拟合曲线
    x_fit = np.linspace(min(x_data), max(x_data), 500)
    y_fit_curve = bessel_model(x_fit, *params)

    # 选择颜色
    color = colors[i % len(colors)]

    # 绘制数据点
    plt.scatter(x_data, y_data, color=color, alpha=0.6)

    # 绘制拟合曲线
    plt.plot(x_fit, y_fit_curve, color=color)

    # **创建分开的 marker 和 line**
    marker_legend = Line2D([0], [0], marker='o', linestyle='None', color=color, markersize=8)
    line_legend = Line2D([0], [0], linestyle='-', color=color)

    # 合并 marker 和 line 成同一图例项
    custom_legend.append((marker_legend, line_legend))
    labels.append(f'{file_name_no_ext} - Data & Fitted')

# 设置刻度字体大小
plt.tick_params(axis='both', which='major', labelsize=18)

# 设置主刻度和次刻度
ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_minor_locator(AutoMinorLocator(5))

# 设置刻度字体大小
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=18)

# 绘制局部线性拟合区间
plt.axvspan(x_min, x_max, color='yellow', alpha=0.3, label='Linear Fit Range')

# 添加轴标签
plt.xlabel('Location (mm)', fontsize=22)
plt.ylabel('Temperature (°C)', fontsize=22)

# **合并 marker 和 linestyle 在一个图例项**
plt.legend(custom_legend, labels, fontsize=18, loc='upper right', ncol=1, handler_map={tuple: HandlerTuple(ndivide=None)})

# 设置坐标范围
plt.xlim(0, 14.5)
plt.ylim(30, 450)

# 显示图表
plt.show()


