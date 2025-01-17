import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np


def create_colorbar(min_value=0,
                    max_value=1,
                    cmap='viridis',
                    orientation='vertical',
                    label='Colorbar Label'):
    """
    生成可自定义颜色范围和配色方案的 colorbar。
    """
    # 1) 创建归一化器
    norm = Normalize(vmin=min_value, vmax=max_value)

    # 2) 创建 ScalarMappable
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # 3) 创建画布和坐标轴
    fig, ax = plt.subplots(figsize=(5, 1) if orientation == 'horizontal' else (1, 5))

    # 4) 添加 colorbar
    cbar = plt.colorbar(sm, cax=ax, orientation=orientation)
    cbar.set_label(label)

    # 5) 展示
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 定义一个示例 cdict，用于更精细地控制 R/G/B 通道
    # 假设我们希望数值范围 [0, 1] 内有多段渐变
    # 注：为了平滑，每个 x 处通常让 left == right
    cdict = {
        'red': [
            (0.0, 0.0, 0.0),  # x=0.0, R通道 = 0.0
            (0.3, 0.0, 0.0),  # x=0.3, 仍然 = 0.0
            (0.7, 1.0, 1.0),  # x=0.7, R通道上升到 1.0
            (1.0, 1.0, 1.0)  # x=1.0, 保持 1.0
        ],
        'green': [
            (0.0, 0.0, 0.0),  # x=0.0, G=0.0
            (0.5, 1.0, 1.0),  # x=0.5, G上升到 1.0
            (1.0, 1.0, 1.0)  # x=1.0, 保持 1.0
        ],
        'blue': [
            (0.0, 1.0, 1.0),  # x=0.0, B=1.0
            (0.5, 0.3, 0.3),  # x=0.5, B=0.3
            (1.0, 0.8, 0.8)  # x=1.0, B=0.8
        ]
    }

    # 用 cdict 创建线性分段 colormap
    my_custom_cmap = LinearSegmentedColormap('my_custom_cmap', cdict)

    # 使用自定义 cmap 生成 colorbar
    create_colorbar(
        min_value=0,
        max_value=100,  # 让 0~100 对应 [0,1] 的渐变
        cmap=my_custom_cmap,
        orientation='vertical',
        label='Temperature(°C)'
    )
