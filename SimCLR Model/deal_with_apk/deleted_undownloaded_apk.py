# -*- coding: utf-8 -*-
#删除androzoo未下载完成的apk，调用的appt工具，使用的绝对路径
#注意加了前面的r，后面的\系统就不会识别为转义字符了
# 指定 APK 文件夹路径
apk_folder = r"D:\2021_malware"

import os
import subprocess


def is_valid_apk(file_path, aapt_path):
    """
    使用 aapt 工具检查 APK 文件的有效性。

    参数:
    - file_path: APK 文件的路径。
    - aapt_path: aapt 可执行文件的绝对路径。

    返回值:
    - 如果 APK 文件有效则返回 True，否则返回 False。
    """
    try:
        # 运行 'aapt dump badging' 命令检查 APK 文件的有效性
        with open(os.devnull, 'w') as devnull:
            subprocess.check_call([aapt_path, 'dump', 'badging', file_path], stdout=devnull, stderr=devnull)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("错误：在指定路径上找不到 'aapt' 命令。")
        return False


def delete_incomplete_apk(directory, aapt_path):
    """
    删除指定目录中不完整的 APK 文件。

    参数:
    - directory: 要搜索不完整 APK 文件的目录。
    - aapt_path: aapt 可执行文件的绝对路径。
    """
    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在。")
        return

    # 列出目录中的所有文件
    files = os.listdir(directory)

    # 遍历每个文件
    for file_name in files:
        file_path = os.path.join(directory, file_name)

        # 检查是否为文件
        if os.path.isfile(file_path):
            # 检查文件是否以 .apk 结尾
            if file_name.endswith('.apk'):
                # 检查 APK 文件是否有效
                if not is_valid_apk(file_path, aapt_path):
                    print(f"删除不完整的 APK 文件: {file_name}")
                    os.remove(file_path)
                    print(f"已删除 {file_name}")


# 指定包含 APK 文件的目录路径
directory_path = apk_folder

# 指定 aapt 可执行文件的绝对路径，安装Androidstudio时自带的
aapt_executable_path = r"C:\Environment\Andriod\SDK\build-tools\30.0.3\aapt.exe"

# 调用函数以删除不完整的 APK 文件
delete_incomplete_apk(directory_path, aapt_executable_path)
