# ReadMe

## English Version

### Project Overview

This project is an implementation of a Convolutional Neural Network (CNN) based on the PyTorch framework for handwritten digit recognition (MNIST dataset). The code includes data preprocessing, model definition, training, evaluation, and model saving functionalities.

### Running Environment

- Python 3.x
- PyTorch
- torchvision

### Dependency Installation

```bash
pip install torch torchvision
```

### Code Structure

- **Data Preprocessing**: Converts the MNIST dataset to Tensor format and normalizes it.
- **Model Definition**: Defines a simple CNN model, including two convolutional layers, two fully connected layers, ReLU activation function, MaxPooling, and Dropout.
- **Training Process**: Uses Adam optimizer and cross-entropy loss function for model training.
- **Model Evaluation**: Evaluates the model's accuracy on the test set and saves the best model.

### Instructions for Use

1. Find the Python file in main repository (`cnn.py`).
2. Run the file in the command line: `python cnn.py`.
3. Observe the loss values during training and the final accuracy on the test set.
4. If the model's accuracy improves, the new model will be automatically saved (in `cnn_mnist.pth`) and the accuracy record will be updated (in `acc.txt`).

### Notes

- The code includes model loading and saving functionalities. If a model has been trained previously, you can continue training or evaluating from the saved model.
- You can adjust the training process by modifying hyperparameters (such as learning rate, batch size, number of training epochs, etc.).
- The current model (stored in 'cnn_mnist.pth') has 9945 corrects out of 10000 images in the evaluation set, i.e. 99.45% accuracy.


## 中文版

### 项目简介

此项目是一个基于PyTorch框架的卷积神经网络（CNN）实现，用于手写数字识别（MNIST数据集）。代码包括了数据预处理、模型定义、训练、评估和模型保存等功能。

### 运行环境

- Python 3.x
- PyTorch
- torchvision

### 依赖安装

```bash
pip install torch torchvision
```

### 代码结构

- **数据预处理**：将MNIST数据集转换为Tensor格式，并进行归一化处理。
- **模型定义**：定义了一个简单的CNN模型，包括两层卷积层、两层全连接层、ReLU激活函数、MaxPooling和Dropout。
- **训练过程**：使用Adam优化器和交叉熵损失函数进行模型训练。
- **模型评估**：在测试集上评估模型的准确率，并保存最佳模型。

### 使用说明

1. 在main中找到该Python文件（`cnn.py`）。
2. 在命令行中运行该文件：`python cnn.py`。
3. 观察训练过程中的损失值和最终测试集上的准确率。
4. 如果模型准确率有所提升，将自动保存新模型（`cnn_mnist.pth`）并更新准确率记录（`acc.txt`）。

### 注意事项

- 代码中包含了模型加载和保存功能，如果之前训练过模型，可以从保存的模型继续训练或评估。
- 可以通过修改超参数（如学习率、批量大小、训练轮数等）来调整训练过程。
- 目前模型（保存在`cnn_mnist.pth`中）在评估集中的10000张图像中有9945识别正确，准确率为99.45%。
