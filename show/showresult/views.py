

from PIL import Image
from django.shortcuts import render

import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn as nn
from .forms import UploadFileForm

def show(request):
    name=""
    result=1
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)

        if form.is_valid():
            # 处理上传的文件，这里简单地将文件保存到服务器上的 media 目录中
            with open('media/' + request.FILES['file'].name, 'wb+') as destination:
                for chunk in request.FILES['file'].chunks():
                    destination.write(chunk)
            name = request.FILES['file'].name
            apkpath='media/' + name
            deal_with_apk(apkpath,name)
            apkpath=apkpath+'_Dex'+'.jpg'
            result=classify(apkpath)
            if result==1:
                result = "恶意软件 "
            else:
                result="不是恶意软件"

    else:
        form = UploadFileForm()
    return render(request, 'show.html', {'form': form, 'result': result,'name':name})
def showresult(request):
    return render(request,'showresult.html')

#!/usr/bin/python
# author zhanghan
# 2024年03月11日
import zipfile

import zlib
import math
import os
import numpy as np
import cv2
def apk2dex(apkpath,apkname):
    """
        将apk中的dex文件提取出来
        :param filepath: apk文件路径
        :return: 命中：True
    """
    # 直接用zipfile.ZipFile处理.apk文件
    apkfile = zipfile.ZipFile(apkpath)
    try:
        outputName=apkname+'_Dex'
        print(outputName)
        outputPath=os.path.join(apkpath+'_Dex')
        print(apkfile.namelist())
        for tempfile in apkfile.namelist():  # 遍历apk包内的所有文件名
            if tempfile.endswith('.dex') :
                f = open(outputPath, 'wb+')
                f.write(apkfile.read(tempfile))
                f.close()
                print('1')
    except zipfile.BadZipFile:
        print('File is not a zip file')
    except zlib.error:
        print('while decompressing data: invalid block type')
    except RuntimeError:
        print('File classes.dex is encrypted, password required for extraction')

    return apkpath+'_Dex'

def deal_with_apk(apkpath,filename):

    dexFiles =apk2dex(apkpath, filename)
    print(dexFiles)
    try:
        r = []
        g = []
        b = []
        index = 0
        # 从文本或二进制文件中的数据构造一个数组。numpy.fromfile(file, dtype=float, count=- 1, sep='', offset=0, *, like=None)
        data = np.fromfile(dexFiles, np.uint8)  # 返回<class 'numpy.ndarray'>
        x = y = int(math.sqrt(len(data)) // 3)
        maxSize = int(x * y * 3)
        # 对之进行切片处理，取从开始到末尾位置的数据
        data = data[:maxSize]
        for i, elem in enumerate(data):
            if i % 3 == 0:
                r.append(elem)
            elif i % 3 == 1:
                g.append(elem)
            else:
                b.append(elem)
        imgData = cv2.merge([np.array(b).reshape(x, y), np.array(g).reshape(x, y), np.array(r).reshape(x, y)])
        cv2.imwrite(dexFiles + ".jpg", imgData)
    except cv2.error:
        print('!_img.empty() in function cv::imwrite')

def classify(img_path):
    # 加载测试集数据


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path)

    img_tensor = transform(img)

    # 加载微调后的SimCLR模型
    class SimCLR(nn.Module):
        def __init__(self, base_encoder, projection_dim=128):
            super(SimCLR, self).__init__()
            self.base_encoder = base_encoder
            self.projection_head = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, projection_dim)
            )

        def forward(self, x):
            h = self.base_encoder(x)
            z = self.projection_head(h)
            return h, z

    # 加载模型
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 加载已经训练好的SimCLR模型
    base_encoder = torchvision.models.resnet18(pretrained=True)
    base_encoder.fc = nn.Identity()
    simclr_model = SimCLR(base_encoder).to(device)
    # 对比没有微调过的
    simclr_model.load_state_dict(torch.load('static/fine_tuned_simclr_model.pth'))

    class Classifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(Classifier, self).__init__()
            self.fc = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            x = self.fc(x)
            return x
    classifier_model = Classifier(512, 2).to(device)
    classifier_model.load_state_dict(torch.load('static/classify_model.pth'))
    simclr_model.eval()
    classifier_model.eval()

    # 在验证集上进行预测并计算评估指标

    with torch.no_grad():
        # 增加batch_size维度
        img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False).to(device)

        features, _ = simclr_model(img_tensor)
        outputs = classifier_model(features)
        _, predicted = torch.max(outputs, 1)
        return predicted[0].item()



