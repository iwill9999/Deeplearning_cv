import matplotlib.pyplot as plt

""""zip 函数
 a = [1,2,3]
 b = [4,5,6]
 zipped = zip(a,b) 打包为元组的列表
 >>> zipped
 [(1,4),(2,5),(3,6)]

"""

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# 随机梯度下降
def gradient(x, y):
    return 2 * x * (x * w - y)


print('Predict(before training)', 4, forward(4))

w_list = []
mse_list = []

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01 * grad
        w_list.append(w)
        print("\tgrad:", x, y, grad)
        l = loss(x, y)
        mse_list.append(l)
    print("progress:", epoch, "w=", w, l)
plt.plot(w_list, mse_list)
plt.ylabel("Loss")
plt.xlabel("w")
plt.show()
