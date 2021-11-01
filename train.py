from transform_tools import *
import datetime
from darknet import *
from tqdm import tqdm


def begin(path1, path2):
    starttime = datetime.datetime.now()

    # 选择内容图片和风格图片
    content_img, w_c, h_c = loading(path1)
    content_img = Variable(content_img).cuda()
    styple_img, w_s, h_s = loading(path2)
    styple_img = Variable(styple_img).cuda()

    # 导入特征提取网络
    cnn = models.vgg16(pretrained=False)
    pre = torch.load("./model/vgg16-397923af.pth")
    cnn.load_state_dict(pre)
    # 提取VGG16的features部分
    cnn = cnn.features
    cnn = cnn.cuda()
    # 重新创建我们的模型
    model = copy.deepcopy(cnn)

    # 提取内容所需要的卷积层
    content_layer = ["Conv_3"]
    # 提取风格所需要的的卷积层
    style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4"]
    # 定义内容和风格的损失值，用于控制内容和风格对最后合成图的影响程度
    content_weight = 1
    style_weight = 1000

    # 存储每一个内容层和风格层的计算图列表
    content_losses = []
    style_losses = []
    # 创建一个空模型，用来存储我们的迁移模型
    new_model = torch.nn.Sequential()

    gram = Gram_matrix()
    new_model = new_model.cuda()
    gram = gram.cuda()
    # 构建迁移模型
    index = 1
    # 从VGG16中提取出前8层来当做风格迁移的特征提取层
    print("模型重构中")
    for layer in tqdm(list(model)[:8]):
        # 如果这一层是我们的Conv2d类型，就将它加入到我们的新模型中
        # if isinstance(layer, BasicBlock):
        if isinstance(layer, torch.nn.Conv2d):
            name = "Conv_" + str(index)
            new_model.add_module(name, layer)
            # 根据选择的层数分配为风格提取和内容提取
            if name in content_layer:
                # 创建特征提取层，
                # 将内容图输入到模型中进行计算，获得卷积得到的feature map
                target = new_model(content_img).clone()
                # 将控制内容图的权重与feature map 构建内容损失函数对象
                content_loss = Content_loss(content_weight, target)
                # 将这个对象添加到模型中
                new_model.add_module("content_loss_" + str(index), content_loss)
                # 记录损失函数计算过程
                content_losses.append(content_loss)

            if name in style_layer:
                # 将内容图输入到模型中进行计算，获得卷积得到的feature map
                target = new_model(styple_img).clone()
                # 将这个feature map构建格拉姆矩阵
                target = gram(target)
                # 实例化风格损失函数
                style_loss = Style_loss(style_weight, target)
                # 讲风格损失函数对象添加到模型中
                new_model.add_module("style_loss_" + str(index), style_loss)
                # 记录损失函数计算过程
                style_losses.append(style_loss)

        if isinstance(layer, torch.nn.ReLU):
            # 激活函数正常激活即可
            name = "Relu_" + str(index)
            new_model.add_module(name, layer)
            index = index + 1

        if isinstance(layer, torch.nn.MaxPool2d):
            # 最大池化层也正常池化即可
            name = "MaxPool_" + str(index)
            new_model.add_module(name, layer)

    # 获取输入图片，避免在卷积过程中对原图产生影响因此使用了clone
    input_img = content_img.clone()
    # 将一个不可训练的tensor类型转化成可训练的Parameter类型，从而使得在训练过程中对input_img进行风格修改
    parameter = torch.nn.Parameter(input_img.data)
    # LBFGS也是一种优化方法，他最大的优点就在于它能够写出一个closure函数，
    # 在函数中记录多个损失函数的计算图，并返回
    # 通过优化器，实现反向传播和参数更新
    # 这里传入的就是训练的图像，由于风格迁移和大多数的深度神经网络训练不大一样
    # 在风格迁移中，我们生成的图像是根据内容图结合风格图，对像素进行优化的
    # 而大部分的深度神经网络是对特征进行提取，这就是为什么我们这里选择LBFGS的原因
    # 在LBFGS优化函数中，我们传入的parameter实际上就是我们图像的像素点，通过优化图像的像素点从而达到风格迁移的目的
    optimizer = torch.optim.LBFGS([parameter])

    epoch_n = 300
    epoch = [0]
    picture_index = 0
    print("图像优化中")
    while epoch[0] <= epoch_n:
        def loss():
            # 初始化梯度，避免梯度叠加
            optimizer.zero_grad()
            style_score = 0
            content_score = 0
            # 将图片的数值（0~255）压缩至0~1
            parameter.data.clamp_(0, 1)
            # 训练模型
            new_model(parameter)
            # 累加风格损失函数和内容损失函数的反向传播值
            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()
            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print("Epoch:{}  Style Loss:{:4f}  Content Loss:{:4f}".format(epoch, style_score.item(),
                                                                              content_score.item()))
            return style_score + content_score

        optimizer.step(loss)

    print("优化结束")
    # torch.save(new_model, "./model/style-0.00007-content-0.000006.pth")
    file_path = "./output/" + "nmsl" + "-" + "nmsl" + ".jpg"
    tensor_picture_save(input_img, file_path, w_c, h_c)
    # img = transform.ToPILImage()
    # # 一般样本不可能只有一个，我们需要把他转化成网格输出
    # img = torchvision.utils.make_grid([content_img.squeeze(0), styple_img.squeeze(0), input_img.squeeze(0)])
    # # 其次，如果我们是在GPU上跑完的结果的话，我们需要再把结果带回cpu才能转化成numpy数组
    # # 但是我们通过上面的size可以知道我们的颜色通道宽高是不对劲的，于是我们需要转变他的维度，就是我们的tran
    # img = img.cpu().numpy().transpose(1, 2, 0)
    # plt.imshow(img)
    #
    # # plt.savefig(file_path)
    # plt.show()
    endtime = datetime.datetime.now()

    print("生成" + "nmsl" + "-" + "nmsl" + ".jpg" + "用时：", (endtime - starttime).seconds, "s")
    return file_path


def gradually_get():
    starttime = datetime.datetime.now()

    # 选择内容图片和风格图片
    content_name = "斯大林"
    style_name = "毕加索"
    content_img, w_c, h_c = loading("./content/" + content_name + ".jpg")
    content_img = Variable(content_img).cuda()
    styple_img, w_s, h_s = loading("./style/" + style_name + ".jpg")
    styple_img = Variable(styple_img).cuda()

    # 导入特征提取网络
    cnn = models.vgg16(pretrained=False)
    pre = torch.load("./model/vgg16-397923af.pth")
    cnn.load_state_dict(pre)
    # 提取VGG16的features部分
    cnn = cnn.features
    cnn = cnn.cuda()
    # 重新创建我们的模型
    model = copy.deepcopy(cnn)

    # 提取内容所需要的卷积层
    content_layer = ["Conv_3"]
    # 提取风格所需要的的卷积层
    style_layer = ["Conv_1", "Conv_2", "Conv_3", "Conv_4"]
    # 定义内容和风格的损失值，用于控制内容和风格对最后合成图的影响程度
    content_weight = 1
    style_weight = 1000

    # 存储每一个内容层和风格层的计算图列表
    content_losses = []
    style_losses = []
    # 创建一个空模型，用来存储我们的迁移模型
    new_model = torch.nn.Sequential()

    gram = Gram_matrix()
    new_model = new_model.cuda()
    gram = gram.cuda()
    # 构建迁移模型
    index = 1
    # 从VGG16中提取出前8层来当做风格迁移的特征提取层
    print("模型重构中")
    for layer in tqdm(list(model)[:8]):
        # 如果这一层是我们的Conv2d类型，就将它加入到我们的新模型中
        # if isinstance(layer, BasicBlock):
        if isinstance(layer, torch.nn.Conv2d):
            name = "Conv_" + str(index)
            new_model.add_module(name, layer)
            # 根据选择的层数分配为风格提取和内容提取
            if name in content_layer:
                # 创建特征提取层，
                # 将内容图输入到模型中进行计算，获得卷积得到的feature map
                target = new_model(content_img).clone()
                # 将控制内容图的权重与feature map 构建内容损失函数对象
                content_loss = Content_loss(content_weight, target)
                # 将这个对象添加到模型中
                new_model.add_module("content_loss_" + str(index), content_loss)
                # 记录损失函数计算过程
                content_losses.append(content_loss)

            if name in style_layer:
                # 将内容图输入到模型中进行计算，获得卷积得到的feature map
                target = new_model(styple_img).clone()
                # 将这个feature map构建格拉姆矩阵
                target = gram(target)
                # 实例化风格损失函数
                style_loss = Style_loss(style_weight, target)
                # 讲风格损失函数对象添加到模型中
                new_model.add_module("style_loss_" + str(index), style_loss)
                # 记录损失函数计算过程
                style_losses.append(style_loss)

        if isinstance(layer, torch.nn.ReLU):
            # 激活函数正常激活即可
            name = "Relu_" + str(index)
            new_model.add_module(name, layer)
            index = index + 1

        if isinstance(layer, torch.nn.MaxPool2d):
            # 最大池化层也正常池化即可
            name = "MaxPool_" + str(index)
            new_model.add_module(name, layer)

    # 获取输入图片，避免在卷积过程中对原图产生影响因此使用了clone
    input_img = content_img.clone()
    # 将一个不可训练的tensor类型转化成可训练的Parameter类型，从而使得在训练过程中对input_img进行风格修改
    parameter = torch.nn.Parameter(input_img.data)
    # LBFGS也是一种优化方法，他最大的优点就在于它能够写出一个closure函数，
    # 在函数中记录多个损失函数的计算图，并返回
    # 通过优化器，实现反向传播和参数更新
    # 这里传入的就是训练的图像，由于风格迁移和大多数的深度神经网络训练不大一样
    # 在风格迁移中，我们生成的图像是根据内容图结合风格图，对像素进行优化的
    # 而大部分的深度神经网络是对特征进行提取，这就是为什么我们这里选择LBFGS的原因
    # 在LBFGS优化函数中，我们传入的parameter实际上就是我们图像的像素点，通过优化图像的像素点从而达到风格迁移的目的
    optimizer = torch.optim.LBFGS([parameter])

    epoch_n = 300
    epoch = [0]
    picture_index = 0
    print("图像优化中")
    while epoch[0] <= epoch_n:
        def loss():
            # 初始化梯度，避免梯度叠加
            optimizer.zero_grad()
            style_score = 0
            content_score = 0
            # 将图片的数值（0~255）压缩至0~1
            parameter.data.clamp_(0, 1)
            # 训练模型
            new_model(parameter)
            # 累加风格损失函数和内容损失函数的反向传播值
            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()
            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print("Epoch:{}  Style Loss:{:4f}  Content Loss:{:4f}".format(epoch, style_score.item(),
                                                                              content_score.item()))
            return style_score + content_score

        path = "./gradually/" + str(picture_index) + ".jpg"
        tensor_picture_save(input_img.clone(), path, w_c, h_c)
        picture_index += 1
        optimizer.step(loss)

    print("优化结束")
    torch.save(new_model, "./model/style-0.00007-content-0.000006.pth")
    # img = transform.ToPILImage()
    # 一般样本不可能只有一个，我们需要把他转化成网格输出
    img = torchvision.utils.make_grid([content_img.squeeze(0), styple_img.squeeze(0), input_img.squeeze(0)])
    # 其次，如果我们是在GPU上跑完的结果的话，我们需要再把结果带回cpu才能转化成numpy数组
    # 但是我们通过上面的size可以知道我们的颜色通道宽高是不对劲的，于是我们需要转变他的维度，就是我们的tran
    img = img.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(img)
    plt.savefig("./output/" + content_name + "-" + style_name + ".jpg")
    plt.show()
    endtime = datetime.datetime.now()

    print("生成" + content_name + "-" + style_name + ".jpg" + "用时：", (endtime - starttime).seconds, "s")


if __name__ == '__main__':
    # name = begin("content/门口.jpg", "style/星空.jpg")
    # gradually_get()
    print()
