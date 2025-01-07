import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

# 定义超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 2

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# 定义卷积神经网络模型
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


# 初始化模型、损失函数和优化器
model = CNN()

current_directory = Path.cwd()
file_path = current_directory / "cnn_mnist.pth"
acc_path = current_directory / "acc.txt"
pre_acc = 0
if file_path.exists():
    model.load_state_dict(torch.load('cnn_mnist.pth'))
    print('Loading existing model...')
    if acc_path.exists():
        with acc_path.open('r') as file:
            pre_acc = float(file.read())
    else:
        pre_acc = 0.99
    print('acc =', pre_acc)
    learning_rate = 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.6f}')

# 测试模型
model.eval()  # 切换到评估模式
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f'Accuracy of the model on the 10000 test images: {100 * acc:.4f}%')

# 保存模型
if acc > pre_acc:
    torch.save(model.state_dict(), 'cnn_mnist.pth')
    file_path = current_directory / 'acc.txt'
    file_path.touch()
    print('Saving better model...')
    # 或者创建并写入文件
    with file_path.open('w') as file:
        file.write(f"{acc:.4f}")
