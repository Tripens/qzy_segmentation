import warnings
import torch.nn as nn
from mmseg.registry import MODELS
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from ..utils import UpConvBlock, Upsample

class BasicConvBlock(nn.Module):
    """
        basic conv block(n × {conv,bn,relu})
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None
                ):
        super().__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out




# 装饰器注册
@MODELS.register_module()
class BASNet(nn.Module):

    def __init__(self, 
                 in_channels=3,
                 base_channels=64,
                 resnet_version="res34",
                 num_stages=7,
                 strides=(1, 1, 1, 1, 1, 1, 1),
                 enc_num_convs=(1, 3, 4, 6, 3, 3, 3),
                 enc_num_channels=(64, 64, 128, 256, 512, 512),
                 enc_dilations=(1, 1, 1, 1, 1, 1, 1),
                 dec_num_convs=(3, 3, 3, 3, 3, 3),
                 dec_num_channels=(512, 512, 256, 128, 64, 64),
                 dec_dilations=(2, 1, 1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='bilinear'),
                 norm_eval=False,
                 dcn=None,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        resnet_versions=["res18","res34","res50"]
        self.resnet_version = resnet_version
        assert resnet_version in resnet_versions, \
            "resnet version must be one of [res18,res34,res50]."

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.num_stages = num_stages
        self.strides = strides
        self.norm_eval = norm_eval
        self.base_channels = base_channels

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # encoder part
        if self.resnet_version == "resnet34":
            # inconv = CBR*1
            inconv = []
            inconv.append(
                BasicConvBlock(
                    in_channels = in_channels,
                    out_channels = enc_num_channels[i],
                    num_convs = enc_num_convs[i],
                    stride = strides[i],
                    dilation = enc_dilations[i],
                    with_cp = with_cp,
                    conv_cfg = conv_cfg,
                    norm_cfg = norm_cfg,
                    act_cfg = act_cfg,
                    dcn=None,
                    plugins=None))
            self.encoder.append(nn.Sequential(*inconv))
            # 4 layers of res34,  = CBR*2*[3,4,6,3]            
            for i in range(1,5):
                resnet = []
                resnet.append(
                    BasicConvBlock(
                        in_channels = enc_num_channels[i-1],
                        out_channels = enc_num_channels[i],
                        num_convs = 2*enc_num_convs[i],
                        stride = strides[i],
                        dilation = enc_dilations[i],
                        with_cp = with_cp,
                        conv_cfg = conv_cfg,
                        norm_cfg = norm_cfg,
                        act_cfg = act_cfg,
                        dcn=None,
                        plugins=None))
                self.encoder.append(nn.Sequential(*resnet))
            # 2 layers with pooling after res34
            for i in range(5,num_stages):   
                enc_conv_block = []
                enc_conv_block.append(
                    nn.MaxPool2d(2,2,ceil_mode=True),
                    BasicConvBlock(
                        in_channels = enc_num_channels[i-1],
                        out_channels = enc_num_channels[i],
                        num_convs = enc_num_convs[i],
                        stride = strides[i],
                        dilation = enc_dilations[i],
                        with_cp = with_cp,
                        conv_cfg = conv_cfg,
                        norm_cfg = norm_cfg,
                        act_cfg = act_cfg,
                        dcn=None,
                        plugins=None))
                self.encoder.append(nn.Sequential(*enc_conv_block))

            # bridging
            bridging = []
            for i in range(3):             
                bridging.append(
                    ConvModule(
                        in_channels = enc_num_channels[-1],
                        out_channels = enc_num_channels[-1],
                        kernel_size = 3,
                        stride = strides[-1],
                        dilation= 2,
                        padding = 2,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            self.encoder.append(nn.Sequential(*bridging))

            # decoder part  
            for i in range(num_stages-1):
                if i == 0:
                    # decoder 6, input=bridging+encoder6
                    dec_conv_block = []             
                    dec_conv_block.append(
                        UpConvBlock(
                            conv_block=BasicConvBlock,
                            in_channels = enc_num_channels[num_stages-1-i],
                            skip_channels = enc_num_channels[num_stages-1-i],
                            out_channels = dec_num_channels[i],
                            num_convs = dec_num_convs[i],
                            stride = strides[i],
                            dilation = dec_dilations[i],
                            with_cp = with_cp,
                            conv_cfg = conv_cfg,
                            norm_cfg = norm_cfg,
                            act_cfg = act_cfg,
                            dcn=None,
                            plugins=None))
                    self.decoder.append(nn.Sequential(*dec_conv_block))

                else:
                    # decoder 5/4/3/2/1, inputs=decoder6/5/4/3/2+encoder5/4/3/2/1
                    dec_conv_block = []             
                    dec_conv_block.append(
                        UpConvBlock(
                            conv_block=BasicConvBlock,
                            in_channels = dec_num_channels[i-1],
                            skip_channels = enc_num_channels[num_stages-1-i],
                            out_channels = dec_num_channels[i],
                            num_convs = dec_num_convs[i],
                            stride = strides[i],
                            dilation = dec_dilations[i],
                            with_cp = with_cp,
                            conv_cfg = conv_cfg,
                            norm_cfg = norm_cfg,
                            act_cfg = act_cfg,
                            dcn=None,
                            plugins=None))
                    self.decoder.append(nn.Sequential(*dec_conv_block))


    def forward(self, x):  # should return a tuple
        self._check_input_divisible(x)
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)# inconv,res34[1,2,3,4],extra layers[5,6] & bridging[7]

        dec_outs = [x]# bridging
        for i in (range(len(self.decoder))):
            x = self.decoder[i](enc_outs[self.num_stages-1-i], x)
            dec_outs.append(x)

        return dec_outs

    def init_weights(self, pretrained=None):
        pass