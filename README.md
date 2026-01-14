基于图像识别的恶意软件检查
# SimCLR Model for Android Malware Detection

## 项目简介

这是一个基于对比学习（SimCLR）的Android恶意软件检测系统。该系统将Android应用程序（APK）的dex文件转换为图像，然后使用SimCLR模型进行特征学习和分类，实现高效的恶意软件检测。

## 项目结构

```
SimCLR Model/
├── __pycache__/                # Python缓存文件
├── deal_with_apk/              # APK处理模块
│   ├── Utils.py                # 工具函数
│   ├── extractDex.py           # 从APK提取dex文件
│   ├── bytes_convert_img.py    # 将dex转换为图像
│   └── ...                     # 其他辅助文件
├── trainDexPics/               # 训练图像数据集（需要用户准备）
├── testDexPics2023/            # 测试图像数据集
├── valDexPics/                 # 验证图像数据集
├── simclrstage1.py             # 第一阶段：对比学习训练
├── simclrstage2.py             # 第二阶段：分类训练
├── classify_and_eval.py        # 模型评估
├── simclr_model.pth            # 预训练的SimCLR模型
├── fine_tuned_simclr_model.pth # 微调后的SimCLR模型
├── classify_model.pth          # 分类器模型
└── *.txt                       # 评估结果文件
```

## 工作流程

1. **数据预处理**
   - 从APK文件中提取dex文件
   - 将dex二进制数据转换为RGB图像
   - 构建训练、测试和验证数据集

2. **对比学习训练**（`simclrstage1.py`）
   - 使用无标签图像数据训练SimCLR模型
   - 应用多种数据扩增技术
   - 学习高质量的图像特征表示

3. **分类训练**（`simclrstage2.py`）
   - 在预训练的SimCLR特征基础上训练分类器
   - 使用有标签数据进行监督学习
   - 实现恶意软件检测功能

4. **模型评估**（`classify_and_eval.py`）
   - 在测试集和验证集上进行分类
   - 计算评估指标（准确率、精确率、召回率、F1分数）
   - 生成评估报告

## 环境依赖

- Python 3.7+
- PyTorch
- torchvision
- numpy
- opencv-python
- scikit-learn

## 使用方法

### 1. 准备数据

#### 1.1 提取dex文件

```bash
cd deal_with_apk
python extractDex.py
```

修改`extractDex.py`中的APK文件夹路径：
```python
# 良性 APK 文件夹路径
benign_apk_path = r'your_benign_apk_path'
# 恶意 APK 文件夹路径
malicious_apk_path = r'your_malicious_apk_path'
```

#### 1.2 将dex转换为图像

```bash
python bytes_convert_img.py
```

修改`bytes_convert_img.py`中的文件路径和处理函数：
```python
# 处理训练集
deal_train()
# 处理测试集
# deal_test()
# 处理验证集
# deal_val()
```

### 2. 训练模型

#### 2.1 第一阶段：对比学习训练

```bash
cd ..
python simclrstage1.py
```

修改`simclrstage1.py`中的配置：
```python
# 数据集路径
data_dir = './trainDexPics'
# 训练轮数
num_epochs = 200
# 批量大小
batch_size = 64
```

#### 2.2 第二阶段：分类训练

```bash
python simclrstage2.py
```

修改`simclrstage2.py`中的配置：
```python
# 有标签数据集路径
labeled_data_dir = './trainDexPics2'
# 训练轮数
num_epochs = 200
```

### 3. 评估模型

```bash
python classify_and_eval.py
```

修改`classify_and_eval.py`中的配置：
```python
# 测试集路径
test_data_dir = './testDexPics2023'
# 验证集路径
val_data_dir = './valDexPics'
# 加载的模型文件
simclr_model.load_state_dict(torch.load('simclrstage2_model.pth'))
classifier_model.load_state_dict(torch.load('classify_model.pth'))
```

## 模型评估

评估结果将保存在以下文件中：
- `classify_results.txt`：测试集分类结果
- `evaluation_results.txt`：测试集评估指标
- `classify_val_results.txt`：验证集分类结果
- `evaluation_val_results.txt`：验证集评估指标

评估指标包括：
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数（F1 Score）

## 注意事项

1. **数据准备**
   - 需要用户自行准备APK数据集
   - 确保APK文件包含`classes.dex`文件
   - 建议将良性和恶意APK分开存放

2. **硬件要求**
   - 推荐使用GPU加速训练
   - 训练大型数据集时需要足够的内存和存储空间

3. **超参数调整**
   - 根据数据集大小调整批量大小和训练轮数
   - 可以尝试不同的数据扩增策略

4. **模型选择**
   - 当前使用ResNet18作为基础编码器
   - 可以尝试其他更复杂的模型（如ResNet50、ResNet101）以提高性能

## 结果说明

- `simclrstage1.py`训练的模型：使用对比学习学习图像特征
- `simclrstage2.py`训练的模型：在预训练特征基础上训练分类器
- 对比学习可以有效提高模型的泛化能力，减少过拟合

## 参考

- SimCLR论文：A Simple Framework for Contrastive Learning of Visual Representations
- Android恶意软件检测相关研究
- PyTorch官方文档

## 许可证

本项目仅供学习和研究使用。

---

**注**：本项目需要用户自行准备训练数据和测试数据。请确保遵循相关法律法规获取和使用APK文件。
