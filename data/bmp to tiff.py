import os
from PIL import Image

def convert_bmp_to_tiff(source_folder, target_folder):
    """
    批量将bmp文件格式转换为tiff格式，并将转换后的文件保存到新的文件夹中。

    :param source_folder: 原始bmp文件所在的文件夹路径
    :param target_folder: 转换后tiff文件保存的文件夹路径
    """
    # 检查目标文件夹是否存在，不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.lower().endswith('.bmp'):
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, os.path.splitext(filename)[0] + '.tiff')

            try:
                # 打开bmp文件并转换为tiff格式
                with Image.open(source_path) as img:
                    img.save(target_path, format='TIFF')
                print(f"成功转换：{filename} -> {os.path.basename(target_path)}")
            except Exception as e:
                print(f"转换失败：{filename}，错误信息：{e}")

if __name__ == "__main__":
    # 指定源文件夹和目标文件夹路径
    source_folder = r"C:\Users\za\Desktop\25V"  # 替换为实际的源文件夹路径
    target_folder = r"C:\Users\za\Desktop\25V\tiff"  # 替换为实际的目标文件夹路径

    # 调用转换函数
    convert_bmp_to_tiff(source_folder, target_folder)
