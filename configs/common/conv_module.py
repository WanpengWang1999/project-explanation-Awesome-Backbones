import torch.nn as nn
from configs.basic.build_layer import build_conv_layer, build_norm_layer, build_activation_layer, build_padding_layer
from core.initialize import kaiming_init, constant_init
import copy

class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').


    ### 卷积块（Conv Block）：打包卷积/标准化/激活层的模块

    该块简化了卷积层的使用，通常与标准化层（例如 BatchNorm）和激活层（例如 ReLU）一起使用。它基于三个构建方法：`build_conv_layer()`、`build_norm_layer()` 和 `build_activation_layer()`。

    此外，本模块还增加了一些附加功能：
    1. 自动设置卷积层的 `bias`。
    2. 支持谱归一化（spectral norm）。
    3. 支持更多的填充模式。在 PyTorch 1.5 之前，nn.Conv2d 仅支持零填充和循环填充，我们添加了 "reflect" 填充模式。

    参数：
    - `in_channels`（int）：输入特征图中的通道数，与 `nn._ConvNd` 中的相同。
    - `out_channels`（int）：卷积操作后得到的通道数，与 `nn._ConvNd` 中的相同。
    - `kernel_size`（int | tuple[int]）：卷积核的大小，与 `nn._ConvNd` 中的相同。
    - `stride`（int | tuple[int]）：卷积的步幅，与 `nn._ConvNd` 中的相同。
    - `padding`（int | tuple[int]）：输入两侧的零填充数，与 `nn._ConvNd` 中的相同。
    - `dilation`（int | tuple[int]）：内核元素之间的间距，与 `nn._ConvNd` 中的相同。
    - `groups`（int）：从输入通道到输出通道的阻塞连接数，与 `nn._ConvNd` 中的相同。
    - `bias`（bool | str）：如果指定为 `auto`，将由 `norm_cfg` 决定。如果 `norm_cfg` 为 `None`，则 `bias` 将设为 True，否则设为 False。默认值为 "auto"。
    - `conv_cfg`（dict）：卷积层的配置字典，默认为 None，表示使用 conv2d。
    - `norm_cfg`（dict）：标准化层的配置字典，默认为 None。
    - `act_cfg`（dict）：激活层的配置字典，默认为 dict(type='ReLU')。
    - `inplace`（bool）：是否对激活使用原地操作模式，默认为 True。
    - `with_spectral_norm`（bool）：是否在卷积模块中使用谱归一化，默认为 False。
    - `padding_mode`（str）：如果当前 PyTorch 中的 `Conv2d` 不支持 `padding_mode`，将使用自定义的填充层。目前支持 ['zeros', 'circular']（官方实现）和 ['reflect']（自定义实现）。默认为 'zeros'。
    - `order`（tuple[str]）：卷积/标准化/激活层的顺序。是 "conv"、"norm" 和 "act" 的序列。常见的示例包括 ("conv", "norm", "act") 和 ("act", "conv", "norm")。默认为 ('conv', 'norm', 'act')。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 inplace=True,
                 with_spectral_norm=False,
                 padding_mode='zeros',
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = copy.deepcopy(conv_cfg)
        self.norm_cfg = copy.deepcopy(norm_cfg)
        self.act_cfg = copy.deepcopy(act_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])
        
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)
        
        conv_padding = 0 if self.with_explicit_padding else padding
        
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
            
        if self.with_norm:
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
            
        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)
            
        self.init_weights()
        
    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None
        
    def init_weights(self):
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)
    
    def forward(self, x, activate=True, norm=True):  
        for layer in self.order:
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)

        return x