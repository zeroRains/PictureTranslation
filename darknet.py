import torch
import torch.nn as nn
import math
from collections import OrderedDict

# 基本的darknet块
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        # 进行两次卷积操作

        # 卷积层
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        # 归一化
        self.bn1 = nn.BatchNorm2d(planes[0])
        # 泄露ReLU
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        # 保存要跳跃链接的x
        residual = x

        # 经过两次卷积激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # 将要跳跃链接的x加到经过两次卷积激活的输出中
        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        # 用于记录当前网络的通道数
        self.inplanes = 32
        # 第一个卷积层将我们的图片从3个通道转化成我们期望的32个通道
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # 归一化
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        #泄露ReLU激活
        self.relu1 = nn.LeakyReLU(0.1)

        # 创建由残差神经网络构成的主干特征提取层，第一个参数用于存放输入和输出的通道数，第二个参数用于表示要存放多少个残差块
        self.layer1 = self._make_layer([32, 64], layers[0])
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            # 如果这一层是卷积层
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        """
        创建主要特征提取层
        :param planes:输入的通道数与输出通道数构成的列表，即上一层的通道数与输出的通道数
        :param blocks:当前的残差网络结构单元创建的个数
        :return:返回创建好的残差网络模型
        """
        # 创建残差网络模型的容器
        layers = []
        # 下采样，步长为2，卷积核大小为3
        # 将这一层的名称与对应的网络结构单元存放在列表中，便于后面的直接添加到空模型中
        # 卷积层：输入当前网络的通道数，输出应该输出的通道数
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False)))
        # 对通道层进行归一化
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        # 对整体进行泄露ReLU激活
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入darknet模块，更新当前的样本通道数
        self.inplanes = planes[1]
        # 根据我们之前说要创建的个数，进行残差块的创建
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        # 将我们的网络层列表转化成有序字典，然后存放到pytorch的空模型中
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # 先进行一次卷积激活，转变图片的通道数
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 进入5个残差网络的主干特征提取
        x = self.layer1(x)
        x = self.layer2(x)
        # 由于后面三个用于上采样，进行预测和分类，于是保留这三个输出
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    # 创建我们的主干特征提取
    model = DarkNet([1, 2, 8, 8, 4])
    # 是否进行预训练
    if pretrained:
        # 你输入的预训练是不是
        if isinstance(pretrained, str):
            # 读取存储好的参数
            model.load_state_dict(torch.load(pretrained))
        else:
            # 路径不对哟
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
