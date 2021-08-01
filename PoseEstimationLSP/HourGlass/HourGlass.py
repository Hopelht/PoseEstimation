# hourglass模型


import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class Bottleneck(nn.Module):  #残差块模型
    expansion = 2  #用于保证后面的相加时的通道数一样

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        # print(inplanes, planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        #k1 s1 p0

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        #k3 s1 p1

        self.bn3 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # print(x)
        residual = x   #残差块的相加的部分

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:  #通道数相同才可以相加
            residual = self.downsample(x)

        out += residual

        return out


# houglass 即自动编码器
class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        # print(block, num_blocks, planes, depth)

        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))

        # print(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))

        # print(hg)
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        # print(n,x)     #n=4
        # print(self.hg[n-1][0](x))

        up1 = self.hg[n - 1][0](x)  #沙漏模型里的上采样

        # print(x.shape)
        low1 = F.max_pool2d(x, 2, stride=2)   #图片尺寸缩小一半

        # print("前",low1.shape)
        low1 = self.hg[n - 1][1](low1)
        # print("后",low1.shape)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)  #递归
        else:
            low2 = self.hg[n - 1][3](low1)

        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)  #上下采样函数

        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        """
            参数解释
            :param block: hg(hourglass)块元素
            :param num_stacks: 有几个hg, num_stacks是表示相同的沙漏堆了多少个
            :param num_blocks: 在每个hg之间有几个block块(残差块)
            :param num_classes: keypoint个数,也就是关节点个数
        """
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)    # 第一次下采样，通道数变为64

        self.bn1 = nn.BatchNorm2d(self.inplanes)   #self.inplanes=64
        # print(self.bn1)  #BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu = nn.ReLU(inplace=True)
        # print(self.relu)
        # print(self.inplanes)

        self.layer1 = self._make_residual(block, self.inplanes, 1) #self.inplanes = 64，有downsample（只是改变channel数）
        # print(self.layer1)  # 残差块
        # print(self.inplanes)
        self.layer2 = self._make_residual(block, self.inplanes, 1) #有downsample（只是改变channel数）
        # print(self.layer2)
        # print(self.inplanes)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        # 这一次的bottleneck没有downsample，因为self.planes == planes(self.num_feats=128)*2 = 256
        # print(self.layer3)
        # print(self.inplanes)

        self.maxpool = nn.MaxPool2d(2, stride=2)#第二次下采样



        # build hourglass modules 构建沙漏模块
        ch = self.num_feats * block.expansion  #128*2=256

        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []

        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            # block, num_blocks=4, planes=128, depth=4
            # print(hg)

            res.append(self._make_residual(block, self.num_feats, num_blocks))

            fc.append(self._make_fc(ch, ch))  #ch=256

            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            # print(score)

            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))

        # print(hg)
        self.hg = nn.ModuleList(hg)
        # print(self.hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        #planes即self.inplanes
        # print(planes)
        # print(blocks)
        downsample = None


        # print(block.expansion)   #即block的变量expansion 2
        # print(self.inplanes, planes * block.expansion)
        if stride != 1 or self.inplanes != planes * block.expansion:
            #将图片通道拉伸至符合block的通道
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        #只在每个block的第一个bottleneck做downsample，因为channel数不相同
        layers.append(block(self.inplanes, planes, stride, downsample))
        # print(block(self.inplanes, planes, stride, downsample))
        # print(layers)


        # print(self.inplanes ,planes)
        self.inplanes = planes * block.expansion



        for i in range(1, blocks):  #当blocks=1 ，后面都不会执行
            # print(blocks)
            # print(tmp)
            layers.append(block(self.inplanes, planes))
            # print(block(self.inplanes, planes))
            # print(layers)
        # print(nn.Sequential(*layers))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):

        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        # print(conv)  # Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))

        return nn.Sequential(
            conv,
            bn,
            self.relu,
        )

    def forward(self, x):
        out = []
        x = self.conv1(x)  #下采样
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)  #下采样
        x = self.layer2(x)
        x = self.layer3(x)


        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            # print(score)

            out.append(score)
            if i < self.num_stacks - 1:

                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_


        # 这里个人加上了连接层方便调整后面预测输出的关节点的形状
        output = out[0]
        output1 = torch.flatten(output,1)
        # print(output1.shape)
        output = nn.Linear(in_features=8192, out_features=28)
        out1 = output(output1)

        return out1






if __name__ == "__main__":
    model = HourglassNet(Bottleneck, num_stacks=2, num_blocks=4, num_classes=2)

    model2 = Hourglass(block=Bottleneck, num_blocks=4, planes=128, depth=4)

    input_data = Variable(torch.rand(2, 3, 256, 256))
    # print(input_data.size())
    # print(input_data)
    input_data2 = Variable(torch.rand(2, 3, 64, 64))

    output = model(input_data2)
    # out = output[0]
    # print(output.shape)   #torch.Size([2, 2, 64, 64])
    # out = torch.flatten(out, 1)
    # output = nn.Linear(in_features=8192, out_features=28)
    # out1 = output(out)

    # print(out.shape)  #torch.Size([2, 8192])
    # print((np.array(output)).shape)
    # writer = SummaryWriter(log_dir='../log', comment='source_arc')
    # with writer:
    #     writer.add_graph(model2, (input_data2, ))

