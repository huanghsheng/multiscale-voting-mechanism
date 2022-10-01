# import torch.nn as nn
# import torch.nn.functional as F
# import math
#
# class FPN(nn.Module):
#     def __init__(self, block, layers):
#         super(FPN, self).__init__()
#         self.inplanes = 64
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # Bottom-up layers
#         self.layer1 = self._make_layer(block,  64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#
#         # Top layer
#         self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
#
#         # Smooth layers
#         self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#
#         # Lateral layers
#         self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample  = None
#         if stride != 1 or self.inplanes != block.expansion * planes:
#             downsample  = nn.Sequential(
#                 nn.Conv2d(self.inplanes, block.expansion * planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(block.expansion * planes)
#             )
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#
#     def _upsample_add(self, x, y):
#         _,_,H,W = y.size()
#         return F.upsample(x, size=(H,W), mode='bilinear') + y
#
#     def forward(self, x):
#         # Bottom-up
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         c1 = self.maxpool(x)
#
#         c2 = self.layer1(c1)
#         c3 = self.layer2(c2)
#         c4 = self.layer3(c3)
#         c5 = self.layer4(c4)
#         # Top-down
#         p5 = self.toplayer(c5)
#         p4 = self._upsample_add(p5, self.latlayer1(c4))
#         p3 = self._upsample_add(p4, self.latlayer2(c3))
#         p2 = self._upsample_add(p3, self.latlayer3(c2))
#         # Smooth
#         p4 = self.smooth1(p4)
#         p3 = self.smooth2(p3)
#         p2 = self.smooth3(p2)
#         return p2, p3, p4, p5



import torch
import torch.nn.functional as F
import torch.nn as nn
class FPN(nn.Module):
    def __init__(self,in_channel_list,out_channel):
        super(FPN, self).__init__()
        self.inner_layer=[]
        self.out_layer=[]
        for in_channel in in_channel_list:
            self.inner_layer.append(nn.Conv2d(in_channel,out_channel,1))
            self.out_layer.append(nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1))
        # self.upsample=nn.Upsample(size=, mode='nearest')
    def forward(self,x):
        head_output=[]
        corent_inner=self.inner_layer[-1](x[-1])
        head_output.append(self.out_layer[-1](corent_inner))
        for i in range(len(x)-2,-1,-1):
            pre_inner=corent_inner
            corent_inner=self.inner_layer[i](x[i])
            size=corent_inner.shape[2:]
            pre_top_down=F.interpolate(pre_inner,size=size)
            add_pre2corent=pre_top_down+corent_inner
            head_output.append(self.out_layer[i](add_pre2corent))
        return list(reversed(head_output))
if __name__ == '__main__':
    fpn=FPN([10,20,30],5)
    x=[]
    x.append(torch.rand(1, 10, 64, 64))
    x.append(torch.rand(1, 20, 16, 16))
    x.append(torch.rand(1, 30, 8, 8))
    c=fpn(x)

    # print(c)