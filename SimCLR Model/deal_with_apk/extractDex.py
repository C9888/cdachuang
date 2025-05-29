import os
import pdb

# 从 Utils 模块导入工具函数
from Utils import *

if __name__ == '__main__':
    index = 0
    # 良性 APK 文件夹路径
    benign_apk_path = r'D:\2023_benign'
    # 恶意 APK 文件夹路径
    malicious_apk_path = r'D:\2023_malware'
    # 获取良性 APK 文件夹中的所有文件名
    #trainPaths = os.listdir(benign_apk_path)
    trainPaths = os.listdir(malicious_apk_path)
    # 如果当前目录下不存在名为 extractedDexs 的文件夹，则创建该文件夹
    if not os.path.isdir("./extractedDexs"):
        os.mkdir("./extractedDexs")
    # 对 trainPaths 中的每个文件进行处理
    for file in trainPaths:
        # 构造 APK 文件的完整路径
        #apkpath = os.path.join(benign_apk_path, file)
        apkpath=os.path.join(malicious_apk_path,file)
        # 将 APK 文件转换为 Dex 格式，并保存到 extractedDexs 文件夹中
        apk2dex(apkpath, file, "./extractedDexs")
    # 打印 "complete!" 表示处理完成
    print("complete!")
