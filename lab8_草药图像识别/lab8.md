# 中山大学计算机学院

人工智能
本科生实验报告

课程名称：Artificial Intelligence

| 学号 | 22336084 | 姓名 | 胡舸耀 |
| ---- | -------- | ---- | ------ |

## 一、实验题目

深度学习——中药图片分类任务

利⽤pytorch框架搭建神经⽹络实现中药图片分类，其中中药图片数据分为训练集train和测试集test，训练集

仅⽤于⽹络训练阶段，测试集仅⽤于模型的性能测试阶段。训练集和测试集均包含五种不同类型的中药图片：

baihe、dangshen、gouqi、huaihua、jinyinhua。请合理设计神经⽹络架构，利⽤训练集完成⽹络训练，统计

⽹络模型的训练准确率和测试准确率，画出模型的训练过程的loss曲线、准确率曲线。

## 二、实验内容

### 1.算法原理

#### （1）CNN神经网络模型

这段代码定义了一个使用PyTorch的神经网络模块（`nn.Module`）来构建卷积神经网络（CNN）的架构。

**类定义** ：

* 定义了 `CNN`类，继承自 `nn.Module`。
* 这是在PyTorch中定义神经网络模型时的常见做法。

**初始化 (`__init__`) 方法** ：

* 在 `__init__` 方法中，构造函数初始化了神经网络的各层。
* 使用 `nn.Conv2d` 定义了三个卷积层 (`conv1`、`conv2`、`conv3`)。具体参数参考代码页
* 每个卷积层具有特定的参数：
  * `in_channels`：输入通道数（设为3，为RGB图像）。
  * `out_channels`：输出通道数或滤波器数。
  * `kernel_size`：卷积核/滤波器的大小。
  * `padding`：卷积时在输入周围添加填充，以保持输入的空间维度。
* 在每个卷积层之后，使用 `nn.MaxPool2d` 定义了最大池化层 (`pool`)。最大池化减少了输入的空间维度，有助于减少计算量和控制过拟合。
* 使用 `nn.Linear` 定义了两个全连接层 (`fc1` 和 `fc2`)。这些层根据卷积层提取的特征进行分类。

**前向传播方法** ：

* `forward` 方法定义了网络的前向传播过程。
* 它指定了输入张量 `x` 如何通过网络的每一层。
* 输入 `x` 通过每个卷积层后，都会经过一个修正线性单元（ReLU）激活函数 (`torch.relu`)。ReLU 引入了非线性到模型中。
* 在经过卷积和池化层之后，张量被重新形状 (`view`) 以与全连接层兼容。
* 重新形状后的张量通过两个全连接层 (`fc1` 和 `fc2`)，产生最终输出。

**输出** ：

* `forward` 方法的输出是最后一个全连接层 (`fc2`) 的输出，它表示每个中草药类别的类别分数。

#### （2）训练器

定义了神经网络具体训练过程

**类定义：**

* 接收参数：
  * `model`: 待训练的神经网络模型。
  * `train_loader`: 训练数据的数据加载器。
  * `test_loader`: 测试数据的数据加载器。
  * `criterion`: 损失函数，用于计算模型输出与目标之间的差异。
  * `optimizer`: 优化器，用于更新模型参数以最小化损失。
  * `num_epochs` (可选): 训练周期数，默认为 10。
* 初始化了一些变量用于跟踪训练和测试过程中的损失和准确率。

**训练方法 :**

* 循环运行指定数量的周期 (`num_epochs`)，在每个周期内调用 `_train_epoch` 和 `_test_epoch` 方法进行训练和测试，并记录损失和准确率。
* 打印每个周期的训练损失、训练准确率、测试损失和测试准确率。

**训练** :

* 将模型置于训练模式 (`self.model.train()`)。
* 对每个训练样本批次执行以下步骤：
  * 将优化器的梯度归零。
  * 通过模型获取预测结果。
  * 计算损失。
  * 反向传播并更新模型参数。
  * 计算并累加该批次的损失。
  * 计算该批次的准确率。
* 返回平均训练损失和训练准确率。

**测试** :

* 将模型置于评估模式 (`self.model.eval()`)。
* 对每个测试样本批次执行以下步骤：
  * 通过模型获取预测结果。
  * 计算损失。
  * 累加该批次的损失。
  * 计算该批次的准确率。
* 返回平均测试损失和测试准确率。其中在测试集上的损失反映了模型对于新数据的拟合程度。如果测试集损失较高，说明模型在未见过的数据上可能泛化能力较差，可能存在过拟合的情况，或者模型的架构、参数等需要进一步优化。因此，测试集上的损失是评估模型性能的一个重要指标之一。

### 2.关键代码展示

- CNN神经网络

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 5)  # 5个中药类别

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

- 训练器

```python
class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_epoch()
            test_loss, test_acc = self._test_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc

    def _test_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_loss = running_loss / len(self.test_loader)
        test_acc = 100. * correct / total
        return test_loss, test_acc
```

- 主函数

```python
def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.ImageFolder(root='train/', transform=transform)
    test_dataset = datasets.ImageFolder(root='test/', transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) #学习率0.001
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, num_epochs=10)
    trainer.train()
    # 绘制训练过程中的损失曲线和准确率曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(trainer.num_epochs), trainer.train_losses, label='Train Loss')
    plt.plot(range(trainer.num_epochs), trainer.test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(trainer.num_epochs), trainer.train_accuracies, label='Train Accuracy')
    plt.plot(range(trainer.num_epochs), trainer.test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
```

## 三、实验结果与分析

#### 1.参数设置

**CNN神经网络参数：**

- 采用简单三层神经网络，每层节点数为（3，32），（32，64），（64，128），最后由两个全连接层归纳到五个药物类别，在传递过程中选择relu激活函数（当权重和小于零时丢失，其他线性激活函数只能表示线性关系，而非线性激活函数可以使神经网络学习到更复杂的模式和特征，这对于图像识别等任务尤其重要。）

**训练参数：**

- 图像转化，将输入的图像大小调整为(224, 224)像素。这是因为许多卷积神经网络（如AlexNet、VGG、ResNet等）在训练时采用了这个固定的输入大小。将图像大小调整为相同的尺寸可以简化网络的设计。`transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`这一步对图像进行标准化处理。
- 训练过程
  - `batch_size=32` 这指定了每个训练批次（batch）中包含的样本数。在训练神经网络时，通常会将数据分成小批次进行处理。（注意：原始测试集文件夹中无类别子文件夹，为了方便数据导入，自行在测试文件夹test/中建立五个类别子文件夹。）
  - `optimizer = optim.Adam(model.parameters(), lr=0.001) #学习率0.001` 当学习率0.001时模型收敛效果较好。
  - `trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, num_epochs=10) `在训练器中，我们将数据分成多段训练，每次训练后检查训练集与测试集的正确率，数据分成十段。当训练到第十次时准确率到达最高值，损失函数趋于平滑

#### 2.实验结果与评测指标展示

结果如下：

![1716433509578](image/lab8/1716433509578.png)

![1716433522538](image/lab8/1716433522538.png)

因为测试数据集样例较少的缘故，测试正确率变化较大，但是可以看到训练数据集正确率在多次训练后逐步上升，最后达到97.89%，同时测试数据集正确率也高于90%取得较优结果

## 四，参考资料

[机器学习算法之——卷积神经网络(CNN)原理讲解](https://zhuanlan.zhihu.com/p/156926543)

[【深度学习】一文弄懂CNN及图像识别(Python)](https://blog.csdn.net/fengdu78/article/details/121882322#:~:text=%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88Convolutional%20Neural,Networks%2C%20CNN%EF%BC%89%E6%98%AF%E4%B8%80%E7%B1%BB%E5%8C%85%E5%90%AB%E5%8D%B7%E7%A7%AF%E8%AE%A1%E7%AE%97%E7%9A%84%E5%89%8D%E9%A6%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%8C%E6%98%AF%E5%9F%BA%E4%BA%8E%E5%9B%BE%E5%83%8F%E4%BB%BB%E5%8A%A1%E7%9A%84%E5%B9%B3%E7%A7%BB%E4%B8%8D%E5%8F%98%E6%80%A7%EF%BC%88%E5%9B%BE%E5%83%8F%E8%AF%86%E5%88%AB%E7%9A%84%E5%AF%B9%E8%B1%A1%E5%9C%A8%E4%B8%8D%E5%90%8C%E4%BD%8D%E7%BD%AE%E6%9C%89%E7%9B%B8%E5%90%8C%E7%9A%84%E5%90%AB%E4%B9%89%EF%BC%89%E8%AE%BE%E8%AE%A1%E7%9A%84%EF%BC%8C%E6%93%85%E9%95%BF%E5%BA%94%E7%94%A8%E4%BA%8E%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86%E7%AD%89%E4%BB%BB%E5%8A%A1%E3%80%82)
