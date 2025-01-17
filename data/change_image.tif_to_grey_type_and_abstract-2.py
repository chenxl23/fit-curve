import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # 用于颜色映射
from matplotlib.colors import Normalize

# RGB 转灰度的加权平均法 #都取平均
def rgb_to_grayscale(image_array):
    return (image_array[:, :, 0] + image_array[:, :, 1] + image_array[:, :, 2])/3

# 批量处理函数
def batch_image_subtraction_with_colormap(background_path, input_folder, output_folder):
    # 读取本底图片
    background_image = Image.open(background_path)
    background_array = rgb_to_grayscale(np.array(background_image))  # 使用加权平均法转为灰度图

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有 TIFF 图片
    for filename in os.listdir(input_folder):
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            # 读取输入图片
            input_image = Image.open(input_path)
            input_array = rgb_to_grayscale(np.array(input_image))  # 使用加权平均法转为灰度图

            # 执行减法
            #difference = np.clip(input_array - background_array, 0, 255)  # 保证像素值范围在0-255 #TODO 这里可以设置为本底-批量图片
            difference = np.clip(input_array - background_array, 0, 255)
            difference_normalized = difference / 255.0  # 归一化到 [0, 1]

            # 应用颜色映射
            colormap = cm.plasma  # 使用 'plasma' 映射，你可以换成其他，如 'viridis', 'inferno'
            colored_difference = colormap(difference_normalized)  # 返回 RGBA 值
            colored_difference = (colored_difference[:, :, :3] * 255).astype(np.uint8)  # 转为RGB格式

            # 保存彩色图
            output_image = Image.fromarray(colored_difference)
            output_path = os.path.join(output_folder, f"{filename}")
            output_image.save(output_path)

    print(f"所有图片已处理完成，结果保存在：{output_folder}")

# 示例用法
background_path = r'C:\Users\za\Desktop\25V\tiff\0sccm 25V×0.2A 4cm.tiff'  # 本底图片路径
input_folder = r'C:\Users\za\Desktop\25V\tiff'  # 输入图片文件夹路径
output_folder = r'C:\Users\za\Desktop\25V\tiff\Processed-1'  # 输出图片文件夹路径

batch_image_subtraction_with_colormap(background_path, input_folder, output_folder)

