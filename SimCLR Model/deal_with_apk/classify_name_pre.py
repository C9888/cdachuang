#!/usr/bin/python
# author zhanghan
# 2024年03月03日

import os
#打标签用
# 指定文件夹路径
folder_path = r"D:\2021_malware"

# 获取文件夹内的所有文件名
file_names = os.listdir(folder_path)

# 遍历文件夹内的所有文件
for i, file_name in enumerate(file_names):
    # 获取文件的扩展名
    ext = os.path.splitext(file_name)[1]
    # 如果文件是遍历指定文件夹中的所有文件，如果文件的扩展名是 .apk，则将其重命名为 malicious_2021_0000.apk 格式的形式，其中 {} 中的数字会逐步递增。
    if ext in ['.apk']:
        # 构造新的文件名
        new_file_name = 'malicious_2021_{:04d}{}'.format(i, ext)
        # 对文件进行重命名
        os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))
