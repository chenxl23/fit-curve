import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def create_colorbar(min_value=0,
                    max_value=1,
                    cmap='viridis',
                    orientation='vertical',
                    label='Colorbar Label'):
    """
    生成可自定义颜色范围和配色方案的colorbar。

    参数：
    -----------
    min_value : float
        颜色映射的最小值
    max_value : float
        颜色映射的最大值
    cmap : str 或 matplotlib.colors.Colormap
        Matplotlib内置的colormap名字，或自定义的Colormap对象
    orientation : str
        颜色条方向，'vertical'或'horizontal'
    label : str
        colorbar的标签
    """
    # 创建归一化器，用于将实际数值映射到[0,1]
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)

    # 创建 ScalarMappable，用于生成色标
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 不需要实际数据，只需要colorbar本身

    # 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(6, 1) if orientation == 'horizontal' else (1, 6))

    # 在当前坐标轴上添加colorbar
    cbar = plt.colorbar(sm, cax=ax, orientation=orientation)

    # 设置颜色条标签
    cbar.set_label(label)

    # 为了让画布紧凑一些，可以加上下面这句
    plt.tight_layout()

    # 显示colorbar
    plt.show()


# 示例用法
if __name__ == "__main__":
    # 生成一个自定义范围在[10, 50]之间，使用"plasma"配色，并且水平放置的颜色条
    create_colorbar(min_value=10,
                    max_value=50,
                    cmap='plasma',
                    orientation='vertical',
                    label='示例Colorbar')
