#!usr/bin/python
# -*- coding: utf-8 -*-
"""
这段代码是一个函数，用于从给定的 APK 文件中提取出 dex 文件。

让我解释一下这个函数的功能和实现：

函数名：apk2dex
参数：
apkpath：APK 文件的路径
outputName：输出文件名
dexpath：输出 dex 文件的目录路径
返回值：如果提取成功，返回 True
功能和实现细节：

函数首先尝试打开给定的 APK 文件，使用 zipfile.ZipFile 对象处理 APK 文件。
然后，它遍历 APK 包内的所有文件名，寻找名为 'classes.dex' 的文件。
如果找到了 'classes.dex' 文件，它将其内容写入到一个新文件中，并保存到指定的输出目录下，新文件的命名是 outputName+'_Dex'。
如果文件不是有效的 zip 文件，会抛出 zipfile.BadZipFile 异常。
如果在解压缩数据时遇到无效的块类型，会抛出 zlib.error 异常。
如果 classes.dex 文件被加密，会抛出 RuntimeError 异常。
这个函数很适合用于从 APK 中提取 dex 文件进行进一步分析和处理。"""
import os
import time
import zipfile
import random
import string
import zlib

#对原始apk文件进行操作，提取dex后进行
def apk2dex(apkpath, outputName, dexpath):
    """
        将apk中的dex文件提取出来
        :param filepath: apk文件路径
        :return: 命中：True
    """
    # 直接用zipfile.ZipFile处理.apk文件
    try:
        apkfile = zipfile.ZipFile(apkpath)
        apkname = apkpath.split(r"/")[-1].split(".")[0]
        outputName=outputName+'_Dex'
        print(outputName)
        outputPath=os.path.join(dexpath,outputName)
        for tempfile in apkfile.namelist():  # 遍历apk包内的所有文件名
            if tempfile == 'classes.dex':
                # dexfilename = ''.join(random.sample(string.ascii_letters + string.digits, 16)) + '.dex'  # 随机16位字符串文件名
                f = open(outputPath, 'wb+')
                f.write(apkfile.read(tempfile))
                f.close()
    except zipfile.BadZipFile:
        print('File is not a zip file')
    except zlib.error:
        print('while decompressing data: invalid block type')
    except RuntimeError:
        print('File classes.dex is encrypted, password required for extraction')

    return True
