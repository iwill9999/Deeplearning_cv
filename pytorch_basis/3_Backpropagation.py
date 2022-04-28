import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 只有一个w 就写一个 要用中括号括起来
w = torch.Tensor([1.0])
w.requires_grad = True  # 计算梯度


def forward(x):
    # 这里的w是Tensor x会自动转换为Tensor
    return x * w


# 上下两个函数都是计算图
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict(before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)

        # 自动把计算链路上所有需要梯度的地方计算出来，然后全存到 w 里，计算图就会被释放
        l.backward()
        print("\tgrad:", x, y, w.grad.item())  # .item()是转换为python 里的标量
        # w.grad 也是一个Tensor 不加.data的话是在构建计算图
        w.data = w.data - 0.01 * w.grad.data

        # 权重里里面的梯度数据清零
        w.grad.data.zero_()

    print("progress:",epoch, l.item())
print("predict (after training)", 4, forward(4).item())
