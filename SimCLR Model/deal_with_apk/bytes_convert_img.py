#!/usr/bin/python
# author zhanghan
# 2024年02月21日
import math
import os
import numpy as np
import cv2
#pip install opencv-python以后直接在conda对应的环境下安装东西https://www.jianshu.com/p/2786224f6bcf

def deal_train():
    # 处理训练集,dex（Dalvik Executable）是Android平台源代码文件（java，kotlin）经过编译、重构、重排、压缩、混淆后的字节码文件，是对传统的class 文件再处理。
    dexPath = "./extractedDexs/"#extract是提取的意思
    # 获取当前路径下的所有文件名
    dexFiles = os.listdir(dexPath)
    if not os.path.isdir("./trainDexPics/"):
        os.mkdir("./trainDexPics/")
    for dexFn in dexFiles:
        try:
            print(dexFn)#输出处理文件名
            r = []
            g = []
            b = []
            index = 0
            #从文本或二进制文件中的数据构造一个数组。numpy.fromfile(file, dtype=float, count=- 1, sep='', offset=0, *, like=None)
            data = np.fromfile(dexPath + dexFn, np.uint8)#返回<class 'numpy.ndarray'>
            x = y = int(math.sqrt(len(data)) // 3)
            maxSize = int(x * y * 3)
            #对之进行切片处理，取从开始到末尾位置的数据
            data = data[:maxSize]
            for i, elem in enumerate(data):
                if i % 3 == 0:
                    r.append(elem)
                elif i % 3 == 1:
                    g.append(elem)
                else:
                    b.append(elem)
            imgData = cv2.merge([np.array(b).reshape(x, y), np.array(g).reshape(x, y), np.array(r).reshape(x, y)])
            cv2.imwrite("./trainDexPics/" + dexFn + ".jpg", imgData)
        except cv2.error:
            print('!_img.empty() in function cv::imwrite')
    print("trianSet complete!")

def deal_test():
    #处理测试集
    dexPath = "./testDexs/"
    dexFiles = os.listdir(dexPath)
    for dexFn in dexFiles:
        try:
            print(dexFn)
            r = []
            g = []
            b = []
            index = 0
            data = np.fromfile(dexPath + dexFn, np.uint8)
            x = y = int(math.sqrt(len(data)) // 3)
            maxSize = int(x * y * 3)
            data = data[:maxSize]
            for i, elem in enumerate(data):
                if i % 3 == 0:
                    r.append(elem)
                elif i % 3 == 1:
                    g.append(elem)
                else:
                    b.append(elem)
            imgData = cv2.merge([np.array(b).reshape(x, y), np.array(g).reshape(x, y), np.array(r).reshape(x, y)])
            cv2.imwrite("./testDexPics/" + dexFn + ".jpg", imgData)
        except cv2.error:
            print('error occurs #################')
    print('test dataset complete')

def deal_val():
    # 处理验证集
    dexPath = "./valDexs/"
    dexFiles = os.listdir(dexPath)
    for dexFn in dexFiles:
        try:
            print(dexFn)
            r = []
            g = []
            b = []
            index = 0
            data = np.fromfile(dexPath + dexFn, np.uint8)
            x = y = int(math.sqrt(len(data)) // 3)
            maxSize = int(x * y * 3)
            data = data[:maxSize]
            for i, elem in enumerate(data):
                if i % 3 == 0:
                    r.append(elem)
                elif i % 3 == 1:
                    g.append(elem)
                else:
                    b.append(elem)
            imgData = cv2.merge([np.array(b).reshape(x, y), np.array(g).reshape(x, y), np.array(r).reshape(x, y)])
            cv2.imwrite("./testDexPics/" + dexFn + ".jpg", imgData)
        except cv2.error:
            print('error occurs #################')
    print('test dataset complete')

if __name__ == "__main__":
    deal_train()
    # deal_test()
    # deal_val()