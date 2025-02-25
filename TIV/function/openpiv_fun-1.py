
import numpy.lib.stride_tricks
import numpy as np
from numpy import ma
from numpy import log





def get_coordinates(image_size, window_size, overlap):
    """
    计算 **光流计算窗口 (interrogation windows) 的中心坐标**。

    该函数返回图像中 **所有窗口的中心点坐标**，用于 **计算运动场的网格化坐标**。

    参数:
    ----------
    image_size : tuple (rows, cols)
        图像大小 (像素)，格式为 (行数, 列数)。
    window_size : int
        窗口大小 (interrogation window)。
    overlap : int
        相邻窗口的 **重叠像素数**。

    返回:
    ----------
    x, y : 2D np.ndarray
        - `x`：包含所有窗口中心点的 **x 坐标** (列坐标)。
        - `y`：包含所有窗口中心点的 **y 坐标** (行坐标)，方向是 **从下到上** (图像坐标)。
    """

    # 计算光流场的形状 (窗口数量)
    field_shape = get_field_shape(image_size, window_size, overlap)

    # 计算所有窗口的中心 x 坐标
    x = (np.arange(field_shape[1]) * (window_size - overlap) +
         (window_size - 1) / 2.0 + ((window_size - overlap) / 2))

    # 计算所有窗口的中心 y 坐标
    y = (np.arange(field_shape[0]) * (window_size - overlap) +
         (window_size - 1) / 2.0 + ((window_size - overlap) / 2))

    # 生成网格坐标，y 方向从下到上排列 ([::-1] 反转 y 方向)
    return np.meshgrid(x, y[::-1])




def get_field_shape(image_size, window_size, overlap):
    """
    计算 **窗口划分后光流场的形状** (行数, 列数)。

    该函数用于计算 **图像被窗口划分后的网格大小**。

    参数:
    ----------
    image_size : tuple (rows, cols)
        图像大小 (像素)，格式为 `(行数, 列数)`。
    window_size : int
        计算窗口的大小 (interrogation window)。
    overlap : int
        相邻窗口的 **重叠像素数**。

    返回:
    ----------
    field_shape : tuple (n_rows, n_cols)
        - `n_rows`：划分后的窗口行数。
        - `n_cols`：划分后的窗口列数。
    """

    return ((image_size[0] - window_size) // (window_size - overlap) + 1,
            (image_size[1] - window_size) // (window_size - overlap) + 1)


def get_org_data(frame_a, search_area_size, overlap):
    """
    计算 **搜索区域的原始数据矩阵 (frame_a_org)**。

    该函数用于 **获取每个窗口中心点的像素值**，用于后续分析。

    参数:
    ----------
    frame_a : 2D np.ndarray
        输入的原始图像 (或帧)。
    search_area_size : int
        搜索区域大小 (搜索窗口)。
    overlap : int
        窗口重叠大小。

    返回:
    ----------
    frame_a_org : 2D np.ndarray
        包含搜索区域中心点像素值的数组，形状为 `(n_rows, n_cols)`。
    """

    # 计算窗口划分后的形状
    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size, overlap)

    # 初始化 frame_a_org 存储中心点数据
    frame_a_org = np.zeros((n_rows, n_cols))

    # 遍历所有窗口
    for k in range(n_rows):
        for m in range(n_cols):
            # 计算当前窗口的索引范围
            il = k * (search_area_size - overlap)  # 左边界 (行)
            ir = il + search_area_size * 0.5       # 计算窗口中心行

            jt = m * (search_area_size - overlap)  # 上边界 (列)
            jb = jt + search_area_size * 0.5       # 计算窗口中心列

            # 取窗口中心像素值
            frame_a_org[k, m] = frame_a[int(ir), int(jb)]

    return frame_a_org

        
import numpy as np

def moving_window_array(array, window_size, overlap):
    """
    该函数使用 **NumPy 的 stride_tricks** 技术 **高效创建滑动窗口数组**。

    目标：
    - 将 2D `array` 变为 **(n_windows, window_size, window_size)** 形状的 3D 数组。
    - 避免 Python **for 循环的高计算开销**，提高 **计算效率**。

    **原理**：
    - 通过 **NumPy 的 stride (步幅)**，不创建数据副本，直接生成新的 **视图 (view)**。
    - 计算 **滑动窗口的位置索引**，并 **直接映射到内存中的原始数组**。

    参数:
    ----------
    array : 2D np.ndarray
        输入的二维数组 (图像)。
    window_size : int
        窗口大小 (interrogation window)。
    overlap : int
        相邻窗口的 **重叠像素数**。

    返回:
    ----------
    3D np.ndarray
        **形状为** `(n_windows, window_size, window_size)` 的滑动窗口数组。
        每个切片 **代表一个窗口**，用于后续计算 **交叉相关 (cross-correlation)**。
    """

    sz = array.itemsize  # 获取数组中单个元素的字节大小 (例如 float32 为 4 字节)
    shape = array.shape  # 记录输入数组的形状 (行, 列)

    # 确保 array 在内存中是 **连续存储** 的，以防 stride 计算出错
    array = np.ascontiguousarray(array)

    # 计算 **步幅 (strides)**：控制窗口如何在内存中移动
    strides = (
        sz * shape[1] * (window_size - overlap),  # 纵向移动步长 (跨行)
        sz * (window_size - overlap),  # 横向移动步长 (跨列)
        sz * shape[1],  # 窗口内行步长
        sz  # 窗口内列步长
    )

    # 计算 **输出数组的形状**
    shape = (
        int((shape[0] - window_size) / (window_size - overlap)) + 1,  # 输出窗口数 (行方向)
        int((shape[1] - window_size) / (window_size - overlap)) + 1,  # 输出窗口数 (列方向)
        window_size,  # 窗口大小 (行)
        window_size   # 窗口大小 (列)
    )

    # 使用 NumPy `stride_tricks` 生成 **滑动窗口**
    return np.lib.stride_tricks.as_strided(array, strides=strides, shape=shape).reshape(-1, window_size, window_size)




def find_first_peak(corr):
    """
    查找 **相关性矩阵 (correlation map) 的最大峰值 (第一峰)**。

    **计算方法**：
    - `np.argmax()` 直接找到 **最大值索引**，避免 `for` 循环，提高计算效率。
    - 通过 `ind // s` 计算 **行索引**，通过 `ind % s` 计算 **列索引**。

    参数:
    ----------
    corr : 2D np.ndarray
        相关性矩阵 (cross-correlation map)。

    返回:
    ----------
    i : int
        第一峰的行索引。
    j : int
        第一峰的列索引。
    corr_max1 : float
        第一峰的相关性值 (最大值)。
    """

    ind = corr.argmax()  # 找到最大值索引 (1D 索引)
    s = corr.shape[1]  # 获取矩阵的列数

    i = ind // s  # 计算行索引
    j = ind % s   # 计算列索引

    return i, j, corr.max()


def find_second_peak(corr, i=None, j=None, width=2):
    """
    查找 **相关性矩阵中的第二峰值 (次大峰值)**。

    **计算方法**：
    - **第一峰值 (i, j) 附近的 `width x width` 区域** 被 **屏蔽**，不参与计算。
    - 在剩余区域中，找到 **第二大的峰值 (次大峰值)**。

    **应用场景**：
    - 用于 **计算光流 (Optical Flow) 或 运动追踪** 时，提取 **多个匹配点**。
    - **提高匹配鲁棒性**，避免单峰误差。

    参数:
    ----------
    corr : 2D np.ndarray
        相关性矩阵 (cross-correlation map)。
    i, j : int (可选)
        第一峰的行、列索引。如果未提供，则先计算 **第一峰**。
    width : int (默认 2)
        **屏蔽区域的半径**，会在 `width x width` 区域屏蔽第一峰周围的值。

    返回:
    ----------
    i : int
        第二峰的行索引。
    j : int
        第二峰的列索引。
    corr_max2 : float
        第二峰的相关性值 (次大峰值)。
    """

    # 如果未提供第一峰值索引，则先计算
    if i is None or j is None:
        i, j, _ = find_first_peak(corr)

    # 创建掩码数组 (Masked Array)
    tmp = corr.view(ma.MaskedArray)

    # 计算屏蔽区域的边界，防止超出矩阵范围
    iini = max(0, i - width)
    ifin = min(i + width + 1, corr.shape[0])
    jini = max(0, j - width)
    jfin = min(j + width + 1, corr.shape[1])

    # 在第一峰值 (i, j) 附近屏蔽 `width x width` 矩阵
    tmp[iini:ifin, jini:jfin] = ma.masked

    # 在屏蔽后的矩阵中查找第二峰值
    i, j, corr_max2 = find_first_peak(tmp)

    return i, j, corr_max2


import numpy as np
from math import log


def find_subpixel_peak_position(corr, subpixel_method='gaussian'):
    """
    计算 **相关性矩阵 (correlation map) 的亚像素级峰值位置**。

    **背景**：
    - 在光流计算、PIV (粒子图像测速) 等应用中，峰值位置通常 **不在整数像素**。
    - 该函数 **通过插值方法** 估计 **更精确的峰值位置**。

    **方法**：
    1. **高斯插值 (gaussian, 默认)**
    2. **质心插值 (centroid, 替代方案)**
    3. **抛物线插值 (parabolic, 替代方案)**

    参数:
    ----------
    corr : 2D np.ndarray
        相关性矩阵 (cross-correlation map)。
    subpixel_method : str, 默认 'gaussian'
        - 'gaussian' (默认)：高斯拟合估计峰值。
        - 'centroid'：质心估计 (如果相关性数据有负值，则自动切换到此方法)。
        - 'parabolic'：抛物线拟合估计峰值。

    返回:
    ----------
    subp_peak_position : tuple (float, float)
        - `row_offset`：峰值在 Y 方向 (行) 的亚像素偏移量。
        - `col_offset`：峰值在 X 方向 (列) 的亚像素偏移量。
    """

    # 默认峰值偏移量为 (0, 0)
    default_peak_position = (0, 0)

    # 获取整数像素级的峰值位置 (行, 列)
    peak1_i, peak1_j, _ = find_first_peak(corr)

    # 如果峰值在矩阵边界，则调整，使其处于 **矩阵内部**
    if peak1_i == 0:
        peak1_i += 1
    if peak1_j == 0:
        peak1_j += 1
    if peak1_i == corr.shape[0] - 1:
        peak1_i -= 1
    if peak1_j == corr.shape[1] - 1:
        peak1_j -= 1

    try:
        # 获取峰值及其邻域像素值
        c = corr[peak1_i, peak1_j]  # 中心点
        cl = corr[peak1_i - 1, peak1_j]  # 左
        cr = corr[peak1_i + 1, peak1_j]  # 右
        cd = corr[peak1_i, peak1_j - 1]  # 下
        cu = corr[peak1_i, peak1_j + 1]  # 上

        # 如果相关性矩阵中有负值，且当前方法为 `gaussian`，则切换到 `centroid`
        if np.any(np.array([c, cl, cr, cd, cu]) < 0) and subpixel_method == 'gaussian':
            subpixel_method = 'centroid'

        try:
            if subpixel_method == 'gaussian':
                # 高斯拟合公式
                subp_peak_position = (
                    peak1_i + ((log(cl) - log(cr)) / (2 * log(cl) - 4 * log(c) + 2 * log(cr))),
                    peak1_j + ((log(cd) - log(cu)) / (2 * log(cd) - 4 * log(c) + 2 * log(cu)))
                )

                # 处理异常情况：如果结果超出范围，则切换到 `centroid`
                if subp_peak_position[0] > corr.shape[0] or subp_peak_position[0] < -corr.shape[0]:
                    subpixel_method = 'centroid'
                if subp_peak_position[1] > corr.shape[1] or subp_peak_position[1] < -corr.shape[1]:
                    subpixel_method = 'centroid'

            if subpixel_method == 'centroid':
                # 质心方法计算
                subp_peak_position = (
                    ((peak1_i - 1) * cl + peak1_i * c + (peak1_i + 1) * cr) / (cl + c + cr),
                    ((peak1_j - 1) * cd + peak1_j * c + (peak1_j + 1) * cu) / (cd + c + cu)
                )

            if subpixel_method == 'parabolic':
                # 抛物线插值方法
                subp_peak_position = (
                    peak1_i + (cl - cr) / (2 * cl - 4 * c + 2 * cr),
                    peak1_j + (cd - cu) / (2 * cd - 4 * c + 2 * cu)
                )

        except:
            # 若计算出错，则返回默认值
            subp_peak_position = default_peak_position

    except IndexError:
        # 若索引越界，则返回默认值
        subp_peak_position = default_peak_position

    # 计算 **亚像素偏移量**
    return subp_peak_position[0] - default_peak_position[0], subp_peak_position[1] - default_peak_position[1]














            




