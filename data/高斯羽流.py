import numpy as np
import matplotlib.pyplot as plt


def satellite_plume(x, y, Q, u, sigma_y, sigma_z, h):
    """
    计算在点(x,y)处的推力器羽流强度（这里简化为浓度）。

    参数:
    x : float or array of floats - 沿着推力方向的距离 (m)
    y : float or array of floats - 垂直于推力方向的距离 (m)
    Q : float - 推力器排放速率 (kg/s 或者其他合适的单位)
    u : float - 推力器喷射速度 (m/s)
    sigma_y : float - 横向扩散参数 (m)
    sigma_z : float - 纵向扩散参数 (m)
    h : float - 推力器喷口高度 (m)

    返回:
    强度或浓度 (任意单位)
    """
    # 注意：这里的公式是基于地面高斯羽流模型改编而来，仅适用于教学目的
    C = (Q / (2 * np.pi * u * sigma_y * sigma_z)) * np.exp(-0.5 * (y ** 2 / sigma_y ** 2)) * (
        np.exp(-0.5 * ((x - h) ** 2 / sigma_z ** 2))
    )
    return C


# 参数设定
Q = 1e-3  # 推力器排放速率 (kg/s)
u = 1000  # 推力器喷射速度 (m/s)
sigma_y = 0.5  # 横向扩散参数 (m)，假定较小值因为是在真空中
sigma_z = 2  # 纵向扩散参数 (m)，同样假定较小值
h = 10.0  # 推力器喷口位置 (m)，设为0表示从原点开始计算

# 创建网格数据
x = np.linspace(0, 50, 200)  # 沿推力方向距离 (m)
y = np.linspace(-10, 10, 200)  # 垂直于推力方向距离 (m)
X, Y = np.meshgrid(x, y)
Z = satellite_plume(X, Y, Q, u, sigma_y, sigma_z, h)

# 绘制图像
plt.figure(figsize=(10, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='plasma')
plt.colorbar(contour, label='Plume Intensity (arbitrary units)')
plt.title('Simplified Satellite Thruster Plume Model')
plt.xlabel('Distance along thrust direction (m)')
plt.ylabel('Distance perpendicular to thrust direction (m)')
plt.grid(True)
plt.show()