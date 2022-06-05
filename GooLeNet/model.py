import torch.nn as nn
import torch
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channel, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channel, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=1, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            BasicConv2d(in_channel, pool, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = [branch1, branch2, branch3, branch4]
        # dim=1 根据channel的深度拼接
        return torch.cat(output, dim=1)


class InceptionAux(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channel, 128, kernel_size=1)  # output[batch, 128, 4, 4]
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N * 512 * 14 * 14 aux2: N * 528 * 14 * 14
        x = self.averagePool(x)
        # aux1: N * 512 * 4 * 4 aux2: N * 528 * 4 * 4
        x = self.conv(x)
        # aux: N * 128 * 4 * 4
        # 从channel的维度开始展平，128*4*4（从128开始乘）
        x = torch.flatten(x, dim=1)
        # 实例化model后可以通过model.train()和model.eval()来控制模型状态
        # model.train()时 self.training =True; model.eval()时 self.training=False
        x = F.dropout(x, 0.5, training=self.training)
        # N * 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N * 1024
        x = self.fc2(x)
        # N * num_classes
        return x


class GooLeNet(nn.Module):
    #                                       是否使用辅助分类器
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GooLeNet, self).__init__()
        self.aux_logits = aux_logits
        # 第一个卷积层输出 112*112*64 输入通道个数3 卷积核个数64
        # (224-7+2*3)/2 +1 =112.5 pytorch默认向下取整
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # 如果池化后得到一个小数 ceil_mode=True 向上取整；False向下取整
        self.max_pool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        # LRN层没什么用先不使用

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        # max_pool
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        # max_pool
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        # max_pool
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        #
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N * 3 * 224 * 224
        x = self.conv1(x)
        # N * 64 * 112 * 112
        x = self.max_pool(x)
        # N * 64 * 56 * 56
        x = self.conv2(x)
        # N * 64 * 56 * 56
        x = self.conv3(x)
        # N * 192 * 56 * 56
        x = self.max_pool(x)

        # N * 192 * 28 * 28
        x = self.inception3a(x)
        # N * 256 * 28 * 28
        x = self.inception3b(x)
        # N * 480 * 28 * 28
        x = self.max_pool(x)
        # N * 480 * 14 * 14
        x = self.inception4a(x)
        # N * 512 * 14 * 14
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N * 832 * 14 * 14
        x = self.max_pool(x)
        # N * 832 * 7 * 7
        x = self.inception5a(x)
        # N * 832 * 7 * 7
        x = self.inception5b(x)
        # N * 1024 * 7 * 7

        x = self.avg_pool(x)
        # N * 1024 * 1 * 1
        x = torch.flatten(x, dim=1)
        # N * 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N * 1000(num_classes)
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mdoe="fan_out", nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

