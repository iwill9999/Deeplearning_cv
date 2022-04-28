import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    #   归一化 均值和标准差
    transforms.Normalize((0.1307,), (0.381,))
])

train_dataset = datasets.MNIST(
    root="E:\\LearningFiles\\dataset\\mnist",
    train=True,
    download=True,
    transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root="E:\\LearningFiles\\dataset\\mnist",
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        #   图像是(N,1,28,28)  -1表示自动计算出N是多少 得到一个N行784列的矩阵
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
        #   第五层直接输出不需要做非线性激活


# 将一轮循环封装成函数
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        #   Forward + Backward + Update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #   每300轮输出一次
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            imgaes, labels = data
            outputs = model(imgaes)
            print('output:', outputs)
            #   取出想两种数据最大的下标 dim=1表示从第一个维度(列)开始查找 一共是10个维度（列）
            #   得出两个参数 ：最大值和最大值的下标
            _, predicted = torch.max(outputs.data, dim=1)
            #   labels是(N,1)的矩阵，N是样本数量,1标记了是哪一个分类(0,1,2,3...9)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy on test set: %d %%' % (100 * correct / total))


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 冲量为0.5优化训练过程

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
