import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn import metrics
import torch.nn as nn
import torch.optim as optim
# 加载测试集数据
test_data_dir = './testDexPics2023'
val_data_dir = './valDexPics'
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
# 加载模型
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

# 加载已经训练好的SimCLR模型
base_encoder = torchvision.models.resnet18(pretrained=True)
base_encoder.fc = nn.Identity()
simclr_model = SimCLR(base_encoder).to(device)
#对比没有微调过的
#simclr_model.load_state_dict(torch.load('fine_tuned_simclr_model.pth'))
#simclr_model.load_state_dict(torch.load('simclr_model.pth'))#只使用第一阶段的没用，丁点儿用都没
simclr_model.load_state_dict(torch.load('simclrstage2_model.pth'))#直接使用第二阶段单独训练出的模型会出现严重的过拟合，只会在2023年数据集训练良好达到99%
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

classifier_model = Classifier(512, len(test_dataset.classes)).to(device)
num_epochs=200
# 定义损失函数（交叉熵损失）
criterion = nn.CrossEntropyLoss()

    # 定义优化器
optimizer = optim.Adam(classifier_model.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            features, _ = simclr_model(images)

        outputs = classifier_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print("epoch=%s"%epoch)
torch.save(classifier_model.state_dict(), 'classify2_model.pth')