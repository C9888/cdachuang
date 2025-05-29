import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn import metrics
import torch.nn as nn
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
    def __init__(self, base_encoder,projection_dim=128):
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
base_encoder = torchvision.models.resnet18(pretrained=True)
base_encoder.fc = nn.Identity()
# 加载已经训练好的SimCLR模型
simclr_model = SimCLR(base_encoder).to(device)
#对比没有微调过的
# simclr_model.load_state_dict(torch.load('fine_tuned_simclr_model.pth'))
simclr_model.load_state_dict(torch.load('simclrstage2_model.pth'))#直接使用第二阶段单独训练出的模型会出现严重的过拟合，只会在2023年数据集训练良好达到99%
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
classifier_model = Classifier(512, len(test_dataset.classes)).to(device)
classifier_model.load_state_dict(torch.load('classify_model.pth'))
# 在测试集上进行预测并计算评估指标
true_labels = []
predicted_labels = []
simclr_model.eval()
classifier_model.eval()
with torch.no_grad(), open("classify_results.txt", "w") as pred_file:
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        features, _ = simclr_model(images)
        outputs = classifier_model(features)

        _, predicted = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

        # 写入预测结果到文件
        for i in range(len(labels)):
            image_label_str = f"Image {i}: Actual - {labels[i].item()}, Predicted - {predicted[i].item()}\n"
            pred_file.write(image_label_str)
pred_file.close()

# 计算评估指标
 # 计算评估指标
 #注意scikit-learn库版本不可太高
accuracy = metrics.accuracy_score(true_labels, predicted_labels)
precision=metrics.precision_score(true_labels, predicted_labels)
recall=metrics.recall_score(true_labels, predicted_labels)
f1_score=metrics.f1_score(true_labels, predicted_labels)

print(accuracy)
print(precision)
print(recall)
print(f1_score)

# 将评估指标写入文件
with open("evaluation_results.txt", "w") as f:
    f.write("Accuracy: {}\n".format(accuracy))
    f.write("Precision: {}\n".format(precision))
    f.write("Recall: {}\n".format(recall))
    f.write("F1 Score: {}\n".format(f1_score))
f.close()
print("测试集评估结果已保存")
# 在验证集上进行预测并计算评估指标
true_labels = []
predicted_labels = []
val_dataset = datasets.ImageFolder(root=val_data_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
with torch.no_grad(), open("classify_val_results.txt", "w") as pred_file:
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        features, _ = simclr_model(images)
        outputs = classifier_model(features)

        _, predicted = torch.max(outputs, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

        # 写入预测结果到文件
        for i in range(len(labels)):
            image_label_str = f"Image {i}: Actual - {labels[i].item()}, Predicted - {predicted[i].item()}\n"
            pred_file.write(image_label_str)
pred_file.close()

# 计算评估指标
 # 计算评估指标
accuracy2 = metrics.accuracy_score(true_labels, predicted_labels)
precision2=metrics.precision_score(true_labels, predicted_labels)
recall2=metrics.recall_score(true_labels, predicted_labels)
f1_score2=metrics.f1_score(true_labels, predicted_labels)
print(accuracy2)
print(precision2)
print(recall2)
print(f1_score2)
# 将评估指标写入文件
with open("evaluation_val_results.txt", "w") as f:
    f.write("Accuracy: {}\n".format(accuracy2))
    f.write("Precision: {}\n".format(precision2))
    f.write("Recall: {}\n".format(recall2))
    f.write("F1 Score: {}\n".format(f1_score2))
f.close()
print("验证集评估结果已保存")