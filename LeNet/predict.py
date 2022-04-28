from model import *
import torch
from PIL import Image
import torchvision.transforms as transform

transforms = transform.Compose(
    [transform.Resize((32, 32)),
     transform.ToTensor(),
     transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('LeNet.pth'))

image = Image.open('./img.png')
image = transforms(image)        # 会得到这样的格式 ： [C, H, W]
#   给图片加一个新的维度Batch
image = torch.unsqueeze(image, dim=0)   # 增加一个维度变成 [M, C, H, W]
#   预测不需要计算梯度
with torch.no_grad():
    outputs = net(image)
    _, predict = torch.max(outputs.data, dim=1)

    #  第0个维度是 batch
    predict2 = torch.softmax(outputs, dim=1)


print(classes[predict])
print(predict2)
