import os
import shutil

def swap_filenames_and_copy(src_dir, dst_dir):
    # 创建目标文件夹，如果不存在的话
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 获取源文件夹中的所有文件
    files = [f for f in os.listdir(src_dir) if f.endswith('.tiff')]

    # 将文件名按数字排序
    files = sorted(files, key=lambda x: float(x.split('.')[0]))

    # 计算需要交换的文件对数
    num_pairs = len(files) // 2

    # 创建一个字典来存储文件名映射
    filename_mapping = {}

    # 生成文件名映射
    for i in range(num_pairs):
        old_name1 = files[i]
        old_name2 = files[-(i + 1)]
        new_name1 = f"{8.9 - float(old_name1.split('.')[0]):.1f}.tiff"
        new_name2 = f"{8.9 - float(old_name2.split('.')[0]):.1f}.tiff"
        filename_mapping[old_name1] = new_name1
        filename_mapping[old_name2] = new_name2

    # 如果文件总数是奇数，中间的文件不需要交换
    if len(files) % 2 == 1:
        middle_file = files[num_pairs]
        filename_mapping[middle_file] = middle_file

    # 复制并重命名文件
    for old_file, new_file in filename_mapping.items():
        old_filepath = os.path.join(src_dir, old_file)
        new_filepath = os.path.join(dst_dir, new_file)
        shutil.copy2(old_filepath, new_filepath)

    print("文件名交换并复制完成。")

# 源目录和目标目录
source_directory = r'E:\数据\20241124\扫描'
destination_directory = r'E:\数据\20241124\扫描\扫描2'

# 调用函数
swap_filenames_and_copy(source_directory, destination_directory)