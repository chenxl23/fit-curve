import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm  # 用于颜色映射

# RGB 转灰度的加权平均法
def rgb_to_grayscale(image_array):
    return (0.2989 * image_array[:, :, 0] +
            0.5870 * image_array[:, :, 1] +
            0.1140 * image_array[:, :, 2])

# 批量处理视频的本底扣除
def video_background_subtraction(background_path, video_path, output_video_path):
    # 读取本底图片
    background_image = Image.open(background_path)
    background_array = rgb_to_grayscale(np.array(background_image))  # 转为灰度图

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)  # 帧率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 帧宽
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧高

    # 定义输出视频文件
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码格式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 读取到视频末尾

        # 将当前帧转为灰度图
        input_array = rgb_to_grayscale(frame)

        # 执行本底扣除
        difference = np.clip(background_array - input_array, 0, 255)
        difference_normalized = difference / 255.0  # 归一化到 [0, 1]

        # 应用颜色映射
        colormap = cm.plasma  # 使用 'plasma' 映射，你可以换成其他，如 'viridis', 'inferno'
        colored_difference = colormap(difference_normalized)  # 返回 RGBA 值
        colored_difference = (colored_difference[:, :, :3] * 255).astype(np.uint8)  # 转为RGB格式

        # 将处理后的帧写入输出视频
        out.write(colored_difference)

    # 释放视频对象
    cap.release()
    out.release()

    print(f"视频处理完成，结果保存在：{output_video_path}")

# 示例用法
background_path = r'C:\Users\za\Desktop\本底降噪.00_00_09_15.Still001.tif'  # 本底图片路径
video_path = r'C:\Users\za\Desktop\本底降噪_1-1.mp4'  # 输入视频路径
output_video_path = r'C:\Users\za\Desktop\本底降噪_000.avi'  # 输出视频路径

video_background_subtraction(background_path, video_path, output_video_path)
