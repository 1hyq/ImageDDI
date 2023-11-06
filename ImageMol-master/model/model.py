import math
import torch.nn as nn
from model.cnn_model_utils import load_model, get_support_model_names

##返回一个3*3的卷积核
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


#定义一个模型，将输入图像转化为嵌入
class ImageMol(nn.Module):
    def __init__(self, baseModel, jigsaw_classes):
        super(ImageMol, self).__init__()

        assert baseModel in get_support_model_names()

        self.baseModel = baseModel

        self.embedding_layer = nn.Sequential(*list(load_model(baseModel).children())[:-1])

        self.bn = nn.BatchNorm1d(512)

        # self.jigsaw_classifier = nn.Linear(512, jigsaw_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
#接受一个输入图像，返回一个特征嵌入
    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.view(x.size(0), -1)
        return x

"""
# to discriminate rationality
class Matcher(nn.Module):
    def __init__(self):
        super(Matcher, self).__init__()
        self.fc = nn.Linear(512, 2)
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

"""

# initializing weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
