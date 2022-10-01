import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding
from utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(400)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # self._conv_head1 = Conv2d(24, 40, kernel_size=1, bias=False)
        self._conv_head2 = Conv2d(40, 80, kernel_size=1, bias=False)
        # self._conv_head3= Conv2d(80, 112, kernel_size=1, bias=False)
        # self._conv_head4 = Conv2d(192, 320, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        # self._bn2 = nn.BatchNorm2d(num_features=40, momentum=bn_mom, eps=bn_eps)
        self._bn2 = nn.BatchNorm2d(num_features=80, momentum=bn_mom, eps=bn_eps)
        # self._bn4 = nn.BatchNorm2d(num_features=112, momentum=bn_mom, eps=bn_eps)
        # self._bn5 = nn.BatchNorm2d(num_features=320, momentum=bn_mom, eps=bn_eps)
        # Final linear layer
        # self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        # Replace AdaptiveAvgPool2d with standard AvgPool2d
        self.avgpool = nn.AvgPool2d((3,3))
        self.maxpool=nn.MaxPool2d((3,3))
        self.avgpool1 = nn.AvgPool2d((7, 7))
        self.maxpool1=nn.MaxPool2d((7,7))
        # self._softmax = F.softmax(dim=1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)

        # self._fcx1= nn.Linear(180000, 5000)
        # self._fcx11 = nn.Linear(5000, 1000)
        # self._fcx12 = nn.Linear(1000, 100)
        # self._fcx13 = nn.Linear(100, 5)

        self._fcx2 = nn.Linear(3360, 500)
        self._fcx21 = nn.Linear(500, 100)
        self._fcx22 = nn.Linear(100, 5)

        self._fcx3 = nn.Linear(840, 100)
        self._fcx31 = nn.Linear(100, 5)

        self._fcx4 = nn.Linear(144, 5)

        # self._fc3 = nn.Linear(15360, 1000)
        self._fc2 = nn.Linear(288, 50)
        self._fc1 = nn.Linear(50, self._global_params.num_classes)

        self._swish = MemoryEfficientSwish()

        # epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
        self.epsilon = 1e-4

        # Conv layers
        self.conv4_up = SeparableConvBlock(24, onnx_export=False)
        self.conv3_up = SeparableConvBlock(24, onnx_export=False)
        self.conv2_up = SeparableConvBlock(24, onnx_export=False)
        self.conv3_down = SeparableConvBlock(24, onnx_export=False)
        self.conv4_down = SeparableConvBlock(24, onnx_export=False)
        self.conv5_down = SeparableConvBlock(24, onnx_export=False)


        # # Feature scaling layers
        # self.p4_upsample = self._upsample_add(x)
        # self.p3_upsample = self._upsample_add(x)
        # self.p2_upsample = self._upsample_add(x)

        # self.p3_downsample = MaxPool2dStaticSamePadding(3, 2)
        # self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        # self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)

        # downSample Lateral layers
        self.downlatlayer = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1)


        self.p5_down_channel = nn.Sequential(
            nn.Conv2d(320, 24, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(24, momentum=0.01, eps=1e-3),
        )

        self.p4_down_channel = nn.Sequential(
            nn.Conv2d(112, 24, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(24, momentum=0.01, eps=1e-3),
        )
        # self.p4_down_channel = nn.Conv2d(112, 24, kernel_size=1, stride=1, padding=0)

        self.p3_down_channel = nn.Sequential(
            nn.Conv2d(40, 24, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(24, momentum=0.01, eps=1e-3),
        )

        self.p2_down_channel = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(24, momentum=0.01, eps=1e-3),
        )

        # Weight
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()

        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()


    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear')

    def _downsample_add(self, x, y):
        x = self.downlatlayer(x)
        _, _, H, W = y.size()
        # x=x[torch.arange(x.size(2))!=1]
        x = x[:, :, :H, :]
        # x=F.interpolate(x, scale_factor=0.5)
        # print(x.size())
        # print(y.size())
        return x

    # 这是为了将特殊的x4的W减一，从而达到特征相加
    def _downsample_add_p5(self, x, y):
        x = self.downlatlayer(x)
        _, _, H, W = y.size()
        # x=x[torch.arange(x.size(2))!=1]
        x = x[:, :, :, :W]
        # x=F.interpolate(x, scale_factor=0.5)
        # print(x.size())
        # print(y.size())
        return x


    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs,bs):
        """ Returns output of the final convolution layer """

        # Stem
        n=0
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        # print(x.size())
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if n==2:
                x2 = x.clone()

                # # x2 = self.avgpool1(x2)
                # x2=self.maxpool1(x2)
                # # print("x2:",x2.size())
                # x2 = x2.view(bs, -1)
                # # x2 = self._dropout(x2)
                # x2 = self._fcx2(x2)
                # x2 = self._fcx21(x2)
                # x2 = F.softmax(x2,dim=1)

            if n==4:
                x3 = x.clone()
                # # x3 = self.avgpool1(x3)
                # x3 = self.maxpool1(x3)
                # # print("x3:",x3.size())
                # # x3 = self._swish(self._bn4(self._conv_head3(x3)))
                # x3 = x3.view(bs, -1)
                # # x3 = self._dropout(x3)
                # x3 = self._fcx3(x3)
                # x3 = self._fcx31(x3)
                #
                # x3 = F.softmax(x3,dim=1)

            # if n == 7:
            #     x4 = x.clone()
                # # x4 = self.avgpool(x4)
                # x4 = self.maxpool(x4)
                # # print("x4:",x4.size())
                # # x4 = self._swish(self._bn5(self._conv_head4(x4)))
                # x4 = x4.view(bs, -1)
                # # x4 = self._dropout(x4)
                # x4 = self._fcx4(x4)
                # x4 = self._fcx41(x4)
                #
                # x4 = F.softmax(x4,dim=1)

            if n==10:
                x4 = x.clone()
                # # x4 = self.avgpool(x4)
                # x4 = self.maxpool(x4)
                # # print("x4:",x4.size())
                # # x4 = self._swish(self._bn5(self._conv_head4(x4)))
                # x4 = x4.view(bs, -1)
                # # x4 = self._dropout(x4)
                # x4 = self._fcx4(x4)
                # x4 = self._fcx41(x4)
                #
                # x4 = F.softmax(x4,dim=1)

            n+=1
            # print(x.size())

        # Head
        # x = self._swish(self._bn1(self._conv_head(x)))

        # return x,x1,x2,x3,x4
        return x, x2, x3, x4
        # return x2
        # return x
    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)

        # Convolution layers
        # x,x1,x2,x3,x4 = self.extract_features(inputs,bs)
        x,x2,x3,x4 = self.extract_features(inputs, bs)

        """
                illustration of a minimal bifpn unit
                    P5_0 -------------------------> P5_2 -------->
                       |-------------|                ↑
                                     ↓                |
                    P4_0 ---------> P4_1 ---------> P4_2 -------->
                       |-------------|--------------↑ ↑
                                     ↓                |
                    P3_0 ---------> P3_1 ---------> P3_2 -------->
                       |-------------|--------------↑ ↑
                                     |--------------↓ |
                    P2_0 -------------------------> P2_2 -------->
                """
        # print(f'x2:{x2.shape}')
        # print(f'x3:{x3.shape}')
        # print(f'x4:{x4.shape}')
        # print(f'x:{x.shape}')

        p2_in = self.p2_down_channel(x2)
        # print(f'p2_in:{p2_in.shape}')
        p3_in = self.p3_down_channel(x3)
        # print(f'p3_in:{p3_in.shape}')
        p4_in = self.p4_down_channel(x4)
        # print(f'p4_in:{p4_in.shape}')
        p5_in = self.p5_down_channel(x)
        # print(f'p5_in:{p5_in.shape}')

        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        # p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_in)))
        # p4_up = (self._swish(weight[0] * p4_in + weight[1] * self._upsample_add(p5_in,p4_in)))
        p4_up =self.conv4_up(self._swish(weight[0] * p4_in + weight[1] * self._upsample_add(p5_in, p4_in)))

        # Weights for P3_0 and P4_1 to P3_1
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        # p3_up = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))
        # p3_up = (self._swish(weight[0] * p3_in + weight[1] * self._upsample_add(p4_up,p3_in)))
        p3_up =self.conv3_up (self._swish(weight[0] * p3_in + weight[1] * self._upsample_add(p4_up, p3_in)))

        # Weights for P2_0 and P3_1 to P2_1
        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        # p2_out = self.conv2_up(self.swish(weight[0] * p2_in + weight[1] * self.p2_upsample(p3_up)))
        # p2_out = (self._swish(weight[0] * p2_in + weight[1] * self._upsample_add(p3_up,p2_in)))
        p2_out = self.conv2_up(self._swish(weight[0] * p2_in + weight[1] * self._upsample_add(p3_up, p2_in)))

        # Weights for P3_0, P3_1 and P2_2 to P3_2
        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        # Connections for P3_0, P3_1 and P2_2 to P3_2 respectively
        # p3_out = self.conv3_down(
        #     self.swish(weight[0] * p3_in + weight[1] * p3_up + weight[2] * self.p3_downsample(p2_out)))
        # p3_out =(
        #     self._swish(weight[0] * p3_in + weight[1] * p3_up + weight[2] * self._downsample_add(p2_out,p3_up)))
        p3_out = self.conv3_down(
            self._swish(weight[0] * p3_in + weight[1] * p3_up + weight[2] * self._downsample_add(p2_out,p3_up)))

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        # p4_out = self.conv4_down(
        #     self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))
        # p4_out =(
        #     self._swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self._downsample_add(p3_out,p4_up)))
        p4_out = self.conv4_down(
            self._swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self._downsample_add(p3_out,p4_up)))

        # Weights for P5_0 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0 and P4_2 to P5_2
        # p5_out = self.conv5_down(self.swish(weight[0] * p5_in + weight[1] * self.p5_downsample(p4_out)))
        # p5_out =(self._swish(weight[0] * p5_in + weight[1] * self._downsample_add_p5(p4_out,p5_in)))
        p5_out = self.conv5_down(self._swish(weight[0] * p5_in + weight[1] * self._downsample_add_p5(p4_out,p5_in)))



        x2=p2_out.clone()
        x2 = self.maxpool1(x2)
        # print("x2",x2.size())
        x2 = x2.view(bs, -1)
        # x2 = self._dropout(x2)
        x2= self._fcx2(x2)
        x2 = self._fcx21(x2)
        x2 = self._fcx22(x2)
        x2 = F.softmax(x2, dim=1)

        x3 = p3_out.clone()
        x3 = self.maxpool1(x3)
        # print("x3", x3.size())
        x3 = x3.view(bs, -1)
        # x3 = self._dropout(x3)
        x3 = self._fcx3(x3)
        x3 = self._fcx31(x3)
        x3 = F.softmax(x3, dim=1)

        x4 = p4_out.clone()
        x4 = self.maxpool1(x4)
        # print("x4", x4.size())
        x4 = x4.view(bs, -1)
        # x4 = self._dropout(x4)
        x4 = self._fcx4(x4)
        x4 = F.softmax(x4, dim=1)

        x = p5_out.clone()
        x = self.maxpool(x)
        # print("x", x.size())
        x = x.view(bs, -1)
        # x = self._dropout(x)
        x = self._fc2(x)
        x = self._fc1(x)
        x = F.softmax(x, dim=1)

        x=x+x2+x3+x4
        x = F.softmax(x, dim=1)
        return x

        # return x,x1,x2,x3,x4
        # return x, x2, x3, x4
        # return p2_out, p3_out, p4_out, p5_out

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b' + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))



# def test():
#     net = EfficientNet.from_name('efficientnet-b0')
#     fms = net(Variable(torch.randn(1,3,300,400)))
#     for fm in fms:
#         print(fm.size())
#
# test()