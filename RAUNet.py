import torch
import torch.nn as nn
from torch.nn import functional as F


# 注意力模块
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# 编码连续卷积层
def contracting_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(kernel_size=(3, 3), in_channels=out_channels, out_channels=out_channels, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels)
    )
    return block


# 上采样过程中也反复使用了两层卷积的结构
double_conv = contracting_block

# 上采样反卷积模块
class expansive_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(expansive_block, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=in_channels, out_channels=out_channels, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        out = self.block(x)
        return out


def final_block(in_channels, out_channels):
    return nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1, padding=0)

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class RAUNet(nn.Module):

    def __init__(self, in_channel, out_channel, block, num_block):
        super().__init__()
        self.in_channels = 64
        # Encode
        #self.conv_encode1 = contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_encode1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#(kernel_size=2, stride=2)
        self.conv_encode2 = self._make_layer(block, 128, num_block[0], 1)
        self.conv_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode3 = self._make_layer(block, 256, num_block[1], 1)
        self.conv_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode4 = self._make_layer(block, 512, num_block[2], 1)
        self.conv_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode5 = self._make_layer(block, 1024, num_block[3], 1)
        #self.conv_encode5 = contracting_block(in_channels=512, out_channels=1024)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Decode
        self.conv_decode4 = expansive_block(1024, 512)
        self.att4 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.double_conv4 = double_conv(1024, 512)

        self.conv_decode3 = expansive_block(512, 256)
        self.att3 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.double_conv3 = double_conv(512, 256)

        self.conv_decode2 = expansive_block(256, 128)
        self.att2 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.double_conv2 = double_conv(256, 128)

        self.conv_decode1 = expansive_block(128, 64)
        self.att1 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.double_conv1 = double_conv(128, 64)

        '''
        self.final_layer = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, out_channel, 1)

        )
        '''
        self.final_layer = final_block(64, out_channel)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_pool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_pool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_pool3(encode_block3)
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_pool4(encode_block4)
        encode_block5 = self.conv_encode5(encode_pool4)

        # Decode
        decode_block4 = self.conv_decode4(encode_block5)
        encode_block4 = self.att4(g=decode_block4, x=encode_block4)
        decode_block4 = torch.cat((encode_block4, decode_block4), dim=1)
        decode_block4 = self.double_conv4(decode_block4)

        decode_block3 = self.conv_decode3(decode_block4)
        encode_block3 = self.att3(g=decode_block3, x=encode_block3)
        decode_block3 = torch.cat((encode_block3, decode_block3), dim=1)
        decode_block3 = self.double_conv3(decode_block3)

        decode_block2 = self.conv_decode2(decode_block3)
        encode_block2 = self.att2(g=decode_block2, x=encode_block2)
        decode_block2 = torch.cat((encode_block2, decode_block2), dim=1)
        decode_block2 = self.double_conv2(decode_block2)

        decode_block1 = self.conv_decode1(decode_block2)
        encode_block1 = self.att1(g=decode_block1, x=encode_block1)
        decode_block1 = torch.cat((encode_block1, decode_block1), dim=1)
        decode_block1 = self.double_conv1(decode_block1)
        #print(decode_block1.shape)
        final_layer = self.final_layer(decode_block1)

        return final_layer


'''
flag = 1

if flag:
    image = torch.rand(1, 3, 256, 256)
    unet = AttUNet(in_channel=3, out_channel=1)
    mask = unet(image)
    print(mask.shape)
'''
def RAUNet34(in_channel,out_channel,pretrain=True):
    """ return a ResNet 34 object
    """
    model=Attruet(in_channel,out_channel,BasicBlock, [3, 4, 6, 3])
    if pretrain:
        model.load_pretrained_weights()
    return model

