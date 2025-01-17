import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# 1. 设置数值范围，这里假设 [0, 100]
vmin, vmax = 20, 25

# 2. 创建归一化器和 colormap
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.get_cmap('plasma')  # 'plasma' 也可以替换成 'viridis'、'inferno' 等

# 3. 构造 ScalarMappable，用于绘制 colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # 不需要实际数据时可以传空

# 4. 创建画布并绘制 colorbar
fig, ax = plt.subplots(figsize=(1, 8))  # 调整宽高，方便展示
cbar = plt.colorbar(sm, cax=ax)

cbar.ax.tick_params(labelsize=12)  #调整colorbar字体

# 5. 设置 colorbar 的标签和标题等
cbar.set_label("Temperature(°C)",fontsize=18)
plt.tight_layout()
plt.show()
