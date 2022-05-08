#   糖尿病分类任务
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(9, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        #   逗号做分隔符 用numpy读取时指定数据类型
        xy = np.loadtxt(filepath, delimiter=' ', dtype=np.float32)
        #    最后一列不要 其余都要 生成Tensor
        self.x_data = torch.from_numpy(xy[:, :-1])
        #   [-1]保证去出来的是一个矩阵
        self.y_data = torch.from_numpy(xy[:, [-1]])
        #   xy 是N行9列的矩阵 shape是(N,9) shape[0]是N 总共的数据样本
        self.len = xy.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes_data.csv.gz')
#   shuffle 打乱数据集顺序 num_workers 读数据集时是不是用并行化 几个并行的进程读取数据
#   batch_size 有多少个batch
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True)

model = Model()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            #   Prepare data
            inputs, labels = data       # 自动转换为Tensor
            #   Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            print(model.linear1.weight.data,model.linear2.weight.data,model.linear3.weight.data)
            #   Backward
            optimizer.zero_grad()
            loss.backward
            #   Update
            optimizer.step()



