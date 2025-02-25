import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib as mpl



# 1. 设置数值范围
vmin, vmax = 23.2,176.7

# 2. 创建归一化器和 colormap
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.get_cmap('inferno')  # plasma / viridis / inferno 等都可以

# 3. 构造 ScalarMappable，用于绘制 colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

# 4. 创建画布并绘制 colorbar
fig, ax = plt.subplots(figsize=(1.5, 12))
cbar = plt.colorbar(sm, cax=ax)

# ------- 核心：自定义 ticks -------
# 方案A：指定“主刻度”的确切位置（含最小值/最大值）
# ticks = [20,40,150]
# cbar.set_ticks(ticks)

# 也可以（可选）自定义显示标签
# cbar.set_ticklabels([f"{t:.1f}" for t in ticks])

# 手动设置 colorbar 的刻度位置，确保 31.1 视觉上居中
cbar.set_ticks([23.2, (23.2 + 176.7) / 2, 176.7])
cbar.set_ticklabels(["23.2", "31.1", "176.7"])  # 设定刻度标签

# 方案B：如果想要固定个数，如 6 等分
# ticks = np.linspace(vmin, vmax, 3)   # 6个等间距刻度
# cbar.set_ticks(ticks)

cbar.ax.tick_params(labelsize=20)  #调整colorbar字体

# 5. 设置 colorbar 的标签等
cbar.set_label("Temperature(°C)",fontsize=22)

# mpl.rcParams['font.size'] = 16  # 16可根据需要调整

# 避免文字或刻度被裁剪
plt.tight_layout()
plt.show()

