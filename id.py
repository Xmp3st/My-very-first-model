import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

# 定义模型（与训练时相同）
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.maxpool(x)
        x = self.relu(self.layer2(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 初始化模型
model = CNN()

# 加载保存的模型状态
file_path = Path.cwd() / "cnn_mnist.pth"
model.load_state_dict(torch.load(file_path, weights_only=False))
model.eval()  # 切换到评估模式

# 定义图像预处理变换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
    transforms.Resize((28, 28)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

# 加载并预处理图像
image_path = './id/base.bmp'  # 替换为你的图像路径
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)  # 添加batch维度

# 进行预测
with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs.data, 1)

# 输出预测结果
print(f'Predicted digit: {predicted.item()}')
