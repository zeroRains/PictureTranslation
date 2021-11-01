import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import copy
import cv2


# 读取图片
def loading(path=None):
    img = Image.open(path)
    width, height = img.size
    img = transform(img)
    # 提高维度
    img = img.unsqueeze(0)
    return img, width, height


# 图像内容损失
class Content_loss(torch.nn.Module):
    # target是通过卷积获取的输入图像中的内容，weight是我们设置的权重参数
    def __init__(self, weight, target):
        super(Content_loss, self).__init__()
        self.weight = weight
        # 用于对提取到的内容进行锁定，不需要进行梯度
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input):
        # input是卷积层中输入的图像
        self.loss = self.loss_fn(input * self.weight, self.target)
        return input

    def backward(self):
        # 保留计算图（中间变量保存下来，便于对style_loss进行计算）
        self.loss.backward(retain_graph=True)
        return self.loss


# 格拉姆矩阵的解析：https://www.cnblogs.com/yifanrensheng/p/12862174.html
# 作用：提取风格图片的风格
class Gram_matrix(torch.nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        gram = gram.div(a * b * c * d)
        return gram


# 图像风格损失
class Style_loss(torch.nn.Module):
    def __init__(self, weight, target):
        super(Style_loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * weight
        self.loss_fn = torch.nn.MSELoss()
        self.gram = Gram_matrix()

    def forward(self, input):
        self.Gram = self.gram(input.clone())
        self.Gram.mul_(self.weight)
        self.loss = self.loss_fn(self.Gram, self.target)
        return input

    def backward(self):
        self.loss.backward(retain_graph=True)
        return self.loss


def tensor_picture_save(img, path, w, h):
    img_temp = img.clone().detach()
    img_temp = img_temp.cpu()
    img_temp = img_temp.squeeze(0)
    img_temp = img_temp.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_RGB2BGR)
    img_temp = cv2.resize(img_temp, (w, h), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path, img_temp)


# 定义转化方式
transform = transforms.Compose([transforms.Resize([300, 300]), transforms.ToTensor()])
