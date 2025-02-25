#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:27:03 2021

@author: benjamin
"""
import numpy as np
from ATIV.functions.openpiv_fun import *
from collections import Counter
from statistics import mode
from PyEMD.EEMD import *
import math
from scipy.signal import hilbert
import copy
import gc
import progressbar

def fill_weight(arr_lst, time_lst):
    arr = np.stack(arr_lst)  # 将 `arr_lst`（列表中的多个数组）堆叠成一个多维 NumPy 数组
    w = np.array(time_lst) / 10  # 将 `time_lst` 转换为 NumPy 数组，并除以 10，得到权重数组
    arr_w = np.average(arr, axis=0, weights=w)  # 使用 `weights=w` 计算加权平均
    return arr_w  # 返回加权平均后的数组


def create_tst_perturbations_mm(array, moving_mean_size=60, showbar=True):
    """
    计算数组的时间移动平均扰动，即每个数据点相对于其局部均值的偏差。

    参数：
    - array: 输入的 NumPy 数组，通常是时间序列数据。
    - moving_mean_size: 移动平均窗口大小（默认 60）。若为 "all"，则窗口大小设为整个数组长度。
    - showbar: 是否显示进度条（默认 True）。

    返回：
    - resultarr: 计算后的扰动数组，每个元素表示原始值减去局部移动均值。
    """

    # 如果移动平均窗口大小设为 "all"，则取整个数组的长度
    if moving_mean_size == "all":
        moving_mean_size = len(array)

    # 初始化结果数组，与输入数组形状相同
    resultarr = np.zeros(np.shape(array))

    # 如果启用了进度条
    if showbar:
        bar = progressbar.ProgressBar(maxval=len(array), widgets=[
            progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()
        ])
        bar.start()
        bar_iterator = 0  # 进度条计数器

    # 遍历输入数组的每个索引 i
    for i in range(len(array)):
        # 选择用于计算移动平均的窗口 actarray
        if i == 0:
            # 处理数组第一个元素，窗口从索引 0 开始，长度为 2*moving_mean_size+1
            actarray = array[0:moving_mean_size*2+1]
        elif i != 0 and i != len(array) and i - moving_mean_size >= 0 and i + moving_mean_size <= len(array) - 1:
            # 处理数组中间的元素，窗口从 i-moving_mean_size 到 i+moving_mean_size
            actarray = array[int(i-moving_mean_size):int(i+moving_mean_size)+1]
        elif i - moving_mean_size <= 0:
            # 处理接近数组开头的情况，窗口固定从 0 开始
            actarray = array[0:moving_mean_size*2+1]
        elif i + moving_mean_size >= len(array):
            # 处理接近数组结尾的情况，窗口从倒数 (2*moving_mean_size+1) 处开始
            actarray = array[len(array)-(2*moving_mean_size)-1:len(array)]
        if i == len(array) - 1:
            # 处理数组最后一个元素，保证窗口大小一致
            actarray = array[len(array)-(2*moving_mean_size)-1:len(array)]

        # 计算扰动值：当前值 - 局部窗口的均值
        resultarr[i] = array[i] - np.mean(actarray, axis=0)

        # 更新进度条
        if showbar:
            bar.update(bar_iterator + 1)
            bar_iterator += 1

    # 结束进度条
    if showbar:
        bar.finish()

    return resultarr

def create_tst_perturbations_spmm(array, moving_mean_size=60, showbar=True):
    """
    计算数组的 **时空移动平均扰动**（Spatiotemporal Moving Mean Perturbation），
    先计算 **空间平均扰动**，然后计算 **时间移动平均扰动**。

    参数：
    - array: 输入的 NumPy 数组（通常是一个 3D 数据，形状为 [时间, X, Y]）。
    - moving_mean_size: 移动均值窗口大小（默认 60）。若为 "all"，则窗口大小设为整个数组长度。
    - showbar: 是否显示进度条（默认 True）。

    返回：
    - resultarr: 计算后的扰动数组，每个元素表示原始值减去 **空间+时间均值**。
    """

    # 如果移动平均窗口大小设为 "all"，则取整个数组的长度
    if moving_mean_size == "all":
        moving_mean_size = len(array)

    # 初始化结果数组，与输入数组形状相同
    resultarr = np.zeros(np.shape(array))

    # 如果启用了进度条
    if showbar:
        bar = progressbar.ProgressBar(maxval=len(array), widgets=[
            progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()
        ])
        bar.start()
        bar_iterator = 0  # 进度条计数器

    # 计算 **空间均值**（忽略 NaN），得到形状 (时间, 1, 1)
    arr_spmean = np.nanmean(array, axis=(1, 2))  # 对 X, Y 维度取均值
    arr_spmean = arr_spmean[:, np.newaxis, np.newaxis]  # 变形，使其广播到原数组形状

    # 计算 **空间扰动**，即减去每个时间步的空间均值
    arr_spperturb = np.ones(array.shape, dtype="int") * arr_spmean  # 复制空间均值到整个数据
    array = array - arr_spperturb  # 减去空间均值，得到去除空间趋势后的数据

    # 遍历时间维度，计算 **时间移动均值扰动**
    for i in range(len(array)):
        # 选择用于计算时间移动平均的窗口 actarray
        if i == 0:
            # 处理时间索引 0 的情况
            actarray = array[0:moving_mean_size*2+1]
        elif i != 0 and i != len(array) and i - moving_mean_size >= 0 and i + moving_mean_size <= len(array) - 1:
            # 处理正常范围内的时间索引
            actarray = array[int(i-moving_mean_size):int(i+moving_mean_size)+1]
        elif i - moving_mean_size <= 0:
            # 处理接近时间起点的情况
            actarray = array[0:moving_mean_size*2+1]
        elif i + moving_mean_size >= len(array):
            # 处理接近时间终点的情况
            actarray = array[len(array)-(2*moving_mean_size)-1:len(array)]
        if i == len(array) - 1:
            # 处理时间最后一个索引
            actarray = array[len(array)-(2*moving_mean_size)-1:len(array)]

        # 计算时间移动均值扰动
        resultarr[i] = array[i] - np.nanmean(actarray, axis=0)

        # 更新进度条
        if showbar:
            bar.update(bar_iterator + 1)
            bar_iterator += 1

    # 结束进度条
    if showbar:
        bar.finish()

    return resultarr


def find_interval(signal, fs, imf_no=1):
    """
    计算最有力的时间间隔 (TIV)。

    该函数基于希尔伯特-黄变换 (HHT) 来分析非平稳时间序列信号。它首先使用 EEMD 分解信号，
    然后对选定的固有模式函数 (IMF) 进行希尔伯特变换，以计算瞬时频率。最后，基于瞬时能量
    计算出最有力的时间间隔。

    参数:
    ----------
    signal : 1D np.ndarray
        一维数组，表示某个像素随时间变化的亮度温度 (扰动)。
    fs : int
        采样频率 (帧率)，即每秒钟记录的帧数。
    imf_no : int, 默认为 1
        要使用的 IMF 编号。IMF 1 具有最高频率。

    返回:
    ----------
    recommended_interval : float
        计算出的最有力的时间间隔 (单位：帧数)，可四舍五入为整数。

    """
    # 1. 经验模态分解 (EEMD) 进行信号分解
    eemd = EEMD()  # 创建 EEMD 对象
    imfs = eemd.eemd(signal)  # 对信号进行 EEMD 分解
    imf = imfs[imf_no - 1, :]  # 选取指定的 IMF 模式 (IMF 1 具有最高频率)

    # 2. 对 IMF 进行 Hilbert 变换，获取瞬时信号
    sig = hilbert(imf)

    # 3. 计算瞬时能量
    energy = np.square(np.abs(sig))

    # 4. 计算瞬时相位
    phase = np.arctan2(sig.imag, sig.real)

    # 5. 计算瞬时频率
    omega = np.gradient(np.unwrap(phase))  # 计算相位梯度，得到瞬时角频率
    omega = fs / (2 * math.pi) * omega  # 转换为 Hz 频率单位

    # 6. 计算加权平均瞬时频率
    insf = omega  # 瞬时频率
    inse = energy  # 瞬时能量
    rel_inse = inse / np.nanmax(inse)  # 归一化瞬时能量 (0-1 之间)
    insf_weighted_mean = np.average(insf, weights=rel_inse)  # 计算加权平均瞬时频率

    # 7. 计算最有力的时间间隔
    insp = 1 / insf_weighted_mean  # 计算最有力的周期 (秒)
    recommended_interval = np.round(fs * insp, 1)  # 转换为帧数，并保留 1 位小数

    # 8. 释放内存
    gc.collect()

    return recommended_interval



def randomize_find_interval(data, rec_freq=1, plot_hht=False, outpath="/", figname="hht_fig"):
    """
    计算最强时间间隔 (TIV) 的统计值。

    该函数是 `find_interval` 的封装版本，它会从数据 `data` 的 **随机像素点** 采样，
    计算最有力的时间间隔 (TIV)，并基于统计模式 (mode) 找到最常出现的时间间隔。

    参数:
    ----------
    data : 3D np.ndarray
        三维数组，表示某个区域的亮度温度（扰动）随时间变化的数据。
    rec_freq : int, 默认 1
        采样频率 (fps)，即数据记录时的帧率。
    plot_hht : bool, 默认 False (尚未实现)
        是否绘制 Hilbert-Huang 变换结果 (未实现)。
    outpath : str, 默认 "/"
        存储绘图的目录路径 (未实现)。
    figname : str, 默认 "hht_fig"
        输出图像的文件名 (未实现)。

    返回:
    ----------
    list
        [第一最常见的间隔, 第二最常见的间隔, 所有计算出的间隔列表]
    """

    masked_boo = True
    for i in range(0, 11):  # 采样 11 组随机像素
        while masked_boo:
            # 1. 随机选取一个像素点 (x, y)
            rand_x = np.round(np.random.rand(), 2)  # 生成 0~1 之间的随机数
            rand_y = np.round(np.random.rand(), 2)

            x = np.round(50 + (225 - 50) * rand_x)  # 将随机数映射到 50~225
            y = np.round(50 + (225 - 50) * rand_y)

            if plot_hht:  # 如果开启调试模式，打印选定的像素坐标
                print(x)
                print(y)

            # 2. 提取该像素的时间序列数据
            pixel = data[:, int(x), int(y)]

            # 3. 如果该像素包含 NaN，则重新随机选择
            if np.isnan(np.sum(pixel)):
                masked_boo = True
            else:
                masked_boo = False

        # 4. 计算该像素的两个 IMF 对应的最有力时间间隔
        act_interval1 = find_interval(pixel, rec_freq, imf_no=1)  # 最高频 IMF
        act_interval2 = find_interval(pixel, rec_freq, imf_no=2)  # 次高频 IMF

        act_intervals = [round(act_interval1), round(act_interval2)]  # 四舍五入的间隔
        act_intervals2 = [act_interval1, act_interval2]  # 原始计算值

        # 5. 存储计算结果
        if i == 0:
            interval_lst = copy.copy([act_intervals])  # 存储整数化的间隔
            interval_lst2 = copy.copy([act_intervals2])  # 存储原始计算值
        else:
            interval_lst.append(act_intervals)
            interval_lst2.append(act_intervals2)

    # 6. 计算最常见的时间间隔 (第一和第二常见)
    try:
        first_most = mode(list(zip(*interval_lst))[0])  # 计算第一个时间间隔的众数
    except:
        # 如果 `mode` 失败，则手动统计出现频次最高的值
        d_same_count_intervals = Counter(list(zip(*interval_lst))[0])
        d_same_count_occ = Counter(d_same_count_intervals.values())

        for value in d_same_count_occ.values():
            if value == 2:
                first_most = np.mean(list(d_same_count_intervals.keys())[0:2])
            if value == 3:
                first_most = np.mean(list(d_same_count_intervals.keys())[0:3])
            if value == 4:
                first_most = np.mean(list(d_same_count_intervals.keys())[0:4])
            if value == 5:
                first_most = np.mean(list(d_same_count_intervals.keys())[0:5])

    try:
        second_most = mode(list(zip(*interval_lst))[1])  # 计算第二个时间间隔的众数
    except:
        sec_most_lst = list(zip(*interval_lst))[1]
        d_same_count_intervals = Counter(sec_most_lst)

        try:
            if first_most in d_same_count_intervals.keys():
                sec_most_lst = np.delete(sec_most_lst, np.where(sec_most_lst == first_most))
                second_most = mode(sec_most_lst)
            else:
                raise ValueError()
        except:
            d_same_count_occ = Counter(d_same_count_intervals.values())
            for value in d_same_count_occ.values():
                if value == 2:
                    second_most = np.mean(list(d_same_count_intervals.keys())[0:2])
                if value == 3:
                    second_most = np.mean(list(d_same_count_intervals.keys())[0:3])
                if value == 4:
                    second_most = np.mean(list(d_same_count_intervals.keys())[0:4])
                if value == 5:
                    second_most = np.mean(list(d_same_count_intervals.keys())[0:5])

    return [first_most, second_most, interval_lst2]



def window_correlation_tiv(frame_a, frame_b, window_size, overlap_window, overlap_search_area, corr_method,
                           search_area_size_x, search_area_size_y=0, window_size_y=0, mean_analysis=True,
                           std_analysis=True, std_threshold=10):
    """
    计算两个图像帧 (frame_a 和 frame_b) 之间的窗口相关性，以确定运动矢量 (TIV)。

    该函数用于分析两个图像帧中的局部运动（光流），采用 **灰度差分、RMSE 或 SSIM** 作为匹配方法。
    计算得到的 u, v 表示每个窗口的位移（速度向量）。

    参数:
    ----------
    frame_a : 2D np.ndarray
        第一帧（基准图像）。
    frame_b : 2D np.ndarray
        第二帧（目标图像）。
    window_size : int
        窗口大小（用于匹配的块大小）。
    overlap_window : int
        窗口重叠量（决定相邻块的重叠度）。
    overlap_search_area : int
        搜索区域重叠量（控制搜索区域的大小）。
    corr_method : str
        相关性计算方法，可选 `"greyscale"`、`"rmse"` 或 `"ssim"`。
    search_area_size_x : int
        搜索区域的宽度（X 方向）。
    search_area_size_y : int, 可选，默认 0
        搜索区域的高度（Y 方向），未使用。
    window_size_y : int, 可选，默认 0
        窗口的高度（Y 方向），未使用。
    mean_analysis : bool, 默认 True
        是否排除 **均值一致的窗口**（避免无信息区域影响计算）。
    std_analysis : bool, 默认 True
        是否排除 **标准差低于阈值的窗口**（避免平坦区域）。
    std_threshold : int, 默认 10
        标准差的阈值，小于此值的区域将被忽略。

    返回:
    ----------
    u, v : 2D np.ndarray
        计算得到的 **X 和 Y 方向的运动矢量**（表示局部位移）。
    """

    # 检查窗口和搜索区域是否满足条件
    if not (window_size - ((search_area_size_x - window_size) / 2)) <= overlap_search_area:
        raise ValueError('搜索区域或窗口重叠度过小，需要满足: ws - (sa - ws)/2 <= ol')

    # 计算图像中可用的窗口数量
    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size_x, overlap_search_area)

    # 初始化位移矩阵
    u = np.zeros((n_rows, n_cols))
    v = np.zeros((n_rows, n_cols))

    # 遍历所有窗口
    for k in range(n_rows):
        for m in range(n_cols):

            # 计算当前窗口在搜索区域中的索引
            il = k * (search_area_size_x - overlap_search_area)  # 左边界 (行方向)
            ir = il + search_area_size_x  # 右边界
            jt = m * (search_area_size_x - overlap_search_area)  # 顶部边界 (列方向)
            jb = jt + search_area_size_x  # 底部边界

            # 从 frame_b 选取搜索区域
            window_b = frame_b[il:ir, jt:jb]

            # 生成滑动窗口数组
            rolling_wind_arr = moving_window_array(window_b, window_size, overlap_window)

            # 计算当前窗口在 frame_a 中的位置
            il += (search_area_size_x - window_size) // 2
            ir = il + window_size
            jt += (search_area_size_x - window_size) // 2
            jb = jt + window_size

            # 从 frame_a 选取匹配窗口
            window_a = frame_a[il:ir, jt:jb]

            # 复制 window_a 以进行计算
            rep_window_a = np.repeat(window_a[:, :, np.newaxis], rolling_wind_arr.shape[0], axis=2)
            rep_window_a = np.rollaxis(rep_window_a, 2)

            # 相关性匹配方法选择
            if corr_method == "greyscale":
                # 计算灰度差异
                dif = rep_window_a - rolling_wind_arr
                dif_sum = np.sum(abs(dif), (1, 2))

                # 重新整形为 2D 相关性矩阵
                shap = int(np.sqrt(rep_window_a.shape[0]))
                dif_sum_reshaped = np.reshape(dif_sum, (shap, shap))
                dif_sum_reshaped = (dif_sum_reshaped * -1) + np.max(dif_sum_reshaped)

                # 计算子像素峰值位置
                row, col = find_subpixel_peak_position(corr=dif_sum_reshaped)

            elif corr_method == "rmse":
                # 计算 RMSE 误差
                rmse = np.sqrt(np.mean((rolling_wind_arr - rep_window_a) ** 2, (1, 2)))

                # 重新整形为 2D 相关性矩阵
                shap = int(np.sqrt(rep_window_a.shape[0]))
                rmse_reshaped = np.reshape(rmse, (shap, shap))
                rmse_reshaped = (rmse_reshaped * -1) + np.max(rmse_reshaped)

                # 计算子像素峰值位置
                row, col = find_subpixel_peak_position(rmse_reshaped)

            elif corr_method == "ssim":
                # 计算结构相似性 (SSIM)
                ssim_lst = ssim(rolling_wind_arr, rep_window_a)

                # 重新整形为 2D 相关性矩阵
                shap = int(np.sqrt(rep_window_a.shape[0]))
                dif_sum_reshaped = np.reshape(ssim_lst, (shap, shap))

                # 计算子像素峰值位置
                row, col = find_subpixel_peak_position(dif_sum_reshaped)

            else:
                raise ValueError("未知的相关性方法，请选择 'greyscale'、'rmse' 或 'ssim'")

            # 归一化偏移量
            row = row - ((shap - 1) / 2)
            col = col - ((shap - 1) / 2)

            # 过滤无信息窗口
            if mean_analysis and np.all(window_a == np.mean(window_a)):
                col = np.nan
                row = np.nan

            if std_analysis and np.std(window_a) < std_threshold:
                col = np.nan
                row = np.nan

            # 存储位移值
            u[k, m], v[k, m] = col, row

    return u, v * -1  # 反转 y 轴方向


def remove_outliers(array, filter_size=5, sigma=1.5):
    """
    该函数用于去除图像或矩阵中的异常值 (outliers)。

    具体方法：
    - 采用 **局部均值 ± sigma * 标准差** 作为正常范围。
    - 计算每个像素的 **局部窗口** (filter_size x filter_size) 的均值和标准差。
    - 如果像素值超出 **上下限**，则替换为 **窗口均值**。

    参数:
    ----------
    array : 2D np.ndarray
        输入的二维数组 (图像)。
    filter_size : int, 默认 5
        滤波窗口大小 (必须是奇数)。
    sigma : float, 默认 1.5
        决定异常值阈值 (标准差倍数)。

    返回:
    ----------
    returnarray : 2D np.ndarray
        去除了异常值的数组 (异常值替换为局部均值)。
    """

    returnarray = copy.copy(array)  # 复制数组，避免修改原始数据
    filter_diff = int(filter_size / 2)  # 计算窗口边界偏移量

    # 初始化进度条
    bar = progressbar.ProgressBar(maxval=array.shape[0], widgets=[
        progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()
    ])
    bar.start()
    bar_iterator = 0  # 进度条计数器

    # 遍历所有像素
    for o in range(array.shape[0]):
        for p in range(array.shape[1]):
            act_px = array[o, p]  # 当前像素值

            try:
                # 计算局部窗口
                act_arr = array[o - filter_diff:o + filter_diff + 1, p - filter_diff:p + filter_diff + 1]

                # 计算局部窗口的均值和标准差 (忽略 NaN)
                mean_val = np.nanmean(act_arr)
                std_val = np.nanstd(act_arr)

                # 计算异常值上下限
                upperlim = mean_val + sigma * std_val
                lowerlim = mean_val - sigma * std_val

                # 如果像素值超出正常范围，则替换为窗口均值
                if act_px < lowerlim or act_px > upperlim:
                    returnarray[o, p] = mean_val
            except:
                pass  # 忽略边界索引异常

        # 更新进度条
        bar.update(bar_iterator + 1)
        bar_iterator += 1

    bar.finish()
    return returnarray


def runTIVparallel(i, interval, perturb, ws, ol, sa, olsa, method, rem_outliers=False, filter_size=3, sigma=2,
                   mean_analysis=False, std_analysis=False, std_threshold=15):
    """
    该函数用于 **并行计算 TIV (Time Interval Velocity, 时间间隔流场)**。

    具体流程：
    1. 计算 **window correlation** 获取局部运动矢量 **(u, v)**。
    2. 若 `rem_outliers=True`，则应用 **去异常值滤波**，去除噪声点。

    参数:
    ----------
    i : int
        当前时间索引 (帧号)。
    interval : int
        计算相邻帧的时间间隔 (TIV)。
    perturb : 3D np.ndarray
        三维扰动数据，形状为 `[时间, X, Y]`。
    ws : int
        相关计算窗口大小 (window size)。
    ol : int
        相关计算窗口的重叠量 (overlap window)。
    sa : int
        搜索区域大小 (search area)。
    olsa : int
        搜索区域重叠量 (overlap search area)。
    method : str
        相关计算方法，可选 `"greyscale"`、`"rmse"` 或 `"ssim"`。
    rem_outliers : bool, 默认 False
        是否 **去除异常值** (离群点)。
    filter_size : int, 默认 3
        用于去异常值的滤波窗口大小 (仅在 `rem_outliers=True` 时生效)。
    sigma : float, 默认 2
        **去异常值的标准差倍数** (仅在 `rem_outliers=True` 时生效)。
    mean_analysis : bool, 默认 False
        是否去除 **均值一致的窗口** (避免无信息区域影响计算)。
    std_analysis : bool, 默认 False
        是否去除 **标准差低于阈值的窗口** (避免平坦区域)。
    std_threshold : int, 默认 15
        标准差的阈值 (仅在 `std_analysis=True` 时生效)。

    返回:
    ----------
    u, v : 2D np.ndarray
        计算得到的 **X 和 Y 方向的运动矢量** (位移场)。
    """

    # 计算窗口相关性，获取 (u, v) 运动矢量
    u, v = window_correlation_tiv(
        frame_a=perturb[i],
        frame_b=perturb[i + interval],
        window_size=ws,
        overlap_window=ol,
        overlap_search_area=olsa,
        search_area_size_x=sa,
        corr_method=method,
        mean_analysis=mean_analysis,
        std_analysis=std_analysis,
        std_threshold=std_threshold
    )

    # 如果需要去除异常值，则对 (u, v) 进行异常值滤波
    if rem_outliers:
        u = remove_outliers(u, filter_size=filter_size, sigma=sigma)
        v = remove_outliers(v, filter_size=filter_size, sigma=sigma)

    return u, v

