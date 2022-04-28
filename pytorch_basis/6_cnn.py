#   每一个通道都要配一个核 输入通道的数量和核的数量是相同的 C必须相同
#   再将卷出来的矩阵相加
#   卷积核的数量就是输出的通道数量 n * w *h 图像 过滤器是 n * w' * h'
#   有m个这样的卷积核 输出   m * wout * hout

import torch

in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size,
                    in_channels,
                    width,
                    height)

conv_layer = torch.nn.Conv2d(in_channels,  # n
                             out_channels,  # m
                             kernel_size=kernel_size)  # 3*3

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)  # 卷积层的维度

input1 = [3, 4, 5, 6, 7,
          5, 4, 6, 4, 1,
          5, 2, 4, 9, 4,
          4, 5, 7, 3, 6,
          7, 5, 1, 2, 9]
input1 = torch.Tensor(input).view(1, 1, 5, 5)  # 变成 B C W H 维度的张量

conv_layer1 = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
#   卷积核
kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)     # O I W H
conv_layer1.weight.data = kernel.data

output1 = conv_layer1(input)
print(output1)

# 池化层   卷积核是2*2
maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)