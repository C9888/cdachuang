import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义SimCLR模型
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.base_encoder = base_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),  # 输入维度为512，输出维度为512的全连接层
            nn.ReLU(),             # ReLU激活函数
            nn.Linear(512, projection_dim)  # 输入维度为512，输出维度为projection_dim的全连接层
        )

    def forward(self, x):
        # 通过基础编码器和投影头进行前向传播
        h = self.base_encoder(x)
        z = self.projection_head(h)
        return h, z

# 定义分类模型
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.classifier_head = nn.Linear(input_dim, num_classes)  # 输入维度为input_dim，输出维度为num_classes的全连接层

    def forward(self, x):
        # 通过分类头进行前向传播
        y = self.classifier_head(x)
        return y

if __name__ == '__main__':
    # 设置设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小为224x224
        transforms.ToTensor(),           # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    # 定义带标签的数据集路径
    labeled_data_dir = './trainDexPics2'

    # 创建数据集
    labeled_dataset = datasets.ImageFolder(root=labeled_data_dir, transform=transform)

    # 定义数据加载器
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=64, shuffle=True, num_workers=8)

    # 定义类别数
    num_classes = len(labeled_dataset.classes)
    # 加载预训练的ResNet18模型
    base_encoder = torchvision.models.resnet18(pretrained=True)
    # 移除ResNet18的分类器层
    base_encoder.fc = nn.Identity()

    # 加载预训练的SimCLR模型
    simclr_model = SimCLR(base_encoder).to(device)
    simclr_model.load_state_dict(torch.load('simclr_model.pth'))

    # 定义分类器模型
    classifier_model = Classifier(512, num_classes).to(device)  # 512是ResNet的最后一层的输出维度

    # 定义损失函数（交叉熵损失）
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = optim.Adam(classifier_model.parameters(), lr=1e-3)

    # 训练模型的总轮数
    num_epochs = 200
    # 迭代训练
    for epoch in range(num_epochs):
        # 将模型设置为训练模式
        classifier_model.train()
        # 遍历带标签数据加载器
        for images, labels in labeled_dataloader:
            # 将图像和标签转移到GPU（如果可用）
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播：通过SimCLR模型提取图像特征
            features, _ = simclr_model(images)
            # 通过分类器模型进行图像分类预测
            outputs = classifier_model(features)

            # 计算损失：利用交叉熵损失函数
            loss = criterion(outputs, labels)

            # 反向传播和优化：清空梯度、计算梯度、更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印当前训练轮数的损失值
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # 保存微调后的simclr模型
    torch.save(simclr_model.state_dict(), 'fine_tuned_simclr_model.pth')
    #torch.save(simclr_model.state_dict(), 'simclrstage2_model.pth')
    torch.save(classifier_model.state_dict(), 'classify_model_2.pth')
    print("Fine-tuning done!")
