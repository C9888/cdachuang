import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# 定义SimCLR模型类
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()
        self.base_encoder = base_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),  # 全连接层，输入维度512，输出维度512
            nn.ReLU(),             # ReLU激活函数
            nn.Linear(512, projection_dim)  # 全连接层，输入维度512，输出维度projection_dim
        )

    def forward(self, x):
        # 通过基础编码器和投影头进行前向传播
        h = self.base_encoder(x)
        z = self.projection_head(h)
        return h, z

if __name__ == '__main__':
    # 设置设备
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 定义数据扩增
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪并调整大小
        transforms.RandomHorizontalFlip(),   # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 随机颜色扭曲
        transforms.RandomRotation(degrees=30),  # 随机旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=0.2),  # 随机仿射变换
        transforms.ToTensor(),               # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    # 定义自定义数据集路径
    data_dir = './trainDexPics'

    # 创建数据集
    custom_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # 定义数据加载器
    train_loader = DataLoader(custom_dataset, batch_size=64, shuffle=True, num_workers=8)

    # 加载预训练的ResNet18模型
    base_encoder = torchvision.models.resnet18(pretrained=True)
    # 移除ResNet18的分类器层
    base_encoder.fc = nn.Identity()

    # 创建SimCLR模型
    simclr_model = SimCLR(base_encoder).to(device)

    # 定义优化器
    optimizer = optim.Adam(simclr_model.parameters(), lr=3e-4)

    # 训练SimCLR模型
    num_epochs = 200

    for epoch in range(num_epochs):
        simclr_model.train()
        total_loss = 0.0
        for images, _ in train_loader:  # 此处只需要图像，标签不需要
            images = images.to(device)

            # 前向传播
            _, projections = simclr_model(images)
            '''计算相似度：
            similarities = torch.mm(projections, projections.t()) / 0.07：首先，我们计算了投影向量的相似度矩阵。
            projections是通过SimCLR模型的投影头产生的投影向量。torch.mm(projections, projections.t())计算了每对投影向量之间的内积，得到一个相似度矩阵。
            然后我们通过除以0.07来调整相似度的尺度，这里的0.07是温度参数。
            计算损失：
            batch_size = similarities.size(0)：获取当前批次的样本数量，即批次大小。
            mask = torch.eye(batch_size, dtype=torch.bool, device=device)：创建一个对角线为True（1），其余元素为False（0）的布尔张量，用于排除同一批次中相同样本之间的相似度计算。
            这样可以确保每个样本与自身的相似度不会被考虑在内。
            pos_pairs = similarities[mask].view(batch_size, -1)：从相似度矩阵中获取同一样本的相似度对，并将其视图重塑为(batch_size, -1)的形状。这里的 -1 表示自动推断维度。
            neg_pairs = similarities[~mask].view(batch_size, -1)：从相似度矩阵中获取不同样本的相似度对，并将其视图重塑为(batch_size, -1)的形状。
            计算交叉熵损失：
            torch.log(torch.exp(pos_pairs / torch.sum(neg_pairs, dim=1, keepdim=True)).sum(dim=1))：这一部分计算了正样本对（pos_pairs）与负样本对（neg_pairs）之间的交叉熵损失。
            首先，我们将负样本对的相似度进行求和并保持其形状与pos_pairs相同。然后，我们将pos_pairs除以这个求和结果，并应用对数函数。最后，我们对这些值进行求和，得到了整个批次的损失。
            .mean()：最后，我们对整个批次的损失值取平均值，得到最终的损失值。
            '''
            # 计算相似度
            similarities = torch.mm(projections, projections.t()) / 0.07  # 计算相似度
            batch_size = similarities.size(0)

            # 计算损失
            mask = torch.eye(batch_size, dtype=torch.bool, device=device)  # 排除同一样本的相似度
            pos_pairs = similarities[mask].view(batch_size, -1)
            neg_pairs = similarities[~mask].view(batch_size, -1)

            # 计算损失，使用的是原作者论文里面的计算
            loss = (torch.log(torch.exp(pos_pairs / torch.sum(neg_pairs, dim=1, keepdim=True)).sum(dim=1))).mean()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        #f 是字符串格式化的一种简便方式，被称为 f-string（formatted string literals）。它允许在字符串中嵌入变量的值和表达式的结果。
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")

    # 保存训练好的模型
    torch.save(simclr_model.state_dict(), 'simclr_model.pth')
