import math
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable
from torchvision.ops import DeformConv2d
from torchvision.ops import deform_conv2d

class DCN(DeformConv2d):
    def __init__(self, *args, **kwargs):
        super(DCN, self).__init__(*args,**kwargs)

        self.deformable_groups = 2
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            2*self.deformable_groups*self.kernel_size[0]*self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=True
        )

        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        assert self.deformable_groups == 2
        offset = self.conv_offset(x)
        return deform_conv2d(input=x, offset=offset, weight=self.weight, stride=_pair(self.stride), padding=_pair(self.padding), dilation=_pair(self.dilation))

if __name__ == '__main__':
    img = torch.rand(32,32,224,224).cuda()
    model = DCN(32,64,3,padding=1).cuda()
    out = model(img)
    print(out.shape)