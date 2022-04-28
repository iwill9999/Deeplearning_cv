import torch

# x_data y_data 都是 3 X 1的矩阵
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


#   Module可以计算方向出传播
#   nn : Neural Network
class LinearModel(torch.nn.Module):
    def __init__(self):     # 构造函数
        super(LinearModel, self).__init__()
        # torch.nn.Linear 是一个类，继承自Module;Linear(1,1) 构造一个对象 包含weight和bias这两个Tensor
        # 自动完成 x*w+b 的计算
        self.linear = torch.nn.Linear(1, 1)  # 两个参数是 输入和输出特征数量 还有一个bias默认是True

    # 重写forward函数
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


#   实例化
model = LinearModel()

#   用yhat和y计算损失函数 求不求平均值都行，没有影响
criterion = torch.nn.MSELoss()

#   优化器 不会构建计算图 model.parameters()会检查model所有的成员,为该实例中可优化的参数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#   训练过程
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

#   所有权重梯度都归零
    optimizer.zero_grad()
#   进行反向传播
    loss.backward()
#   更新
    optimizer.step()

#   Output weight and bias
print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

#   Test Model
x_test = torch.Tensor([4.0])
y_test = model(x_test)
print('y_pred =', y_test.data)

