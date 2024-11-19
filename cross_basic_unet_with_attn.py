
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep

__all__ = ["CrossBasicUNet"]


class TwoConv(nn.Sequential):
    """two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()

        conv_0 = Convolution(spatial_dims, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(
            spatial_dims, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
    ):
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()
        max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCatAllWithAttention(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        encoder_channels: List[int],
        out_chns: int,
        act: str | tuple,
        norm: str | tuple,
        bias: bool,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
        halves: bool = True,
    ):
        super().__init__()
        if upsample == "nontrainable":
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            spatial_dims,
            in_chns,
            up_chns,
            2,
            mode=upsample,
        )
        self.encoder_channels = encoder_channels
        total_cat_channels = up_chns + sum(encoder_channels)
        self.attention = ChannelAttention(total_cat_channels)  # 新增通道注意力模块
        self.convs = TwoConv(spatial_dims, total_cat_channels, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, encoder_features: List[torch.Tensor]):
        x_up = self.upsample(x)
        
        # 调整编码器特征的尺寸以匹配当前解码器层
        resized_encoder_features = []
        for enc_feat in encoder_features:
            resized_feat = torch.nn.functional.interpolate(enc_feat, size=x_up.shape[2:], mode='nearest')
            resized_encoder_features.append(resized_feat)
        
        # 拼接上采样特征和所有编码器特征
        x_cat = torch.cat([x_up] + resized_encoder_features, dim=1)

        # 应用注意力机制
        x_cat = self.attention(x_cat)

        # 卷积融合
        x = self.convs(x_cat)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, num_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction, num_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        b, c, _, _ = x.size()
        # 通道权重计算
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CrossBasicUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 2,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        fea = ensure_tuple_rep(features, 6)
        
        # 编码器部分
        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout)
        
        # 编码器通道列表
        encoder_channels = [fea[0], fea[1], fea[2], fea[3], fea[4]]
        
        # 解码器部分，使用修改后的 UpCatAllWithAttention 模块
        self.upcat_4 = UpCatAllWithAttention(spatial_dims, fea[4], encoder_channels, fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCatAllWithAttention(spatial_dims, fea[3], encoder_channels, fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCatAllWithAttention(spatial_dims, fea[2], encoder_channels, fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCatAllWithAttention(spatial_dims, fea[1], encoder_channels, fea[5], act, norm, bias, dropout, upsample, halves=False)
        
        self.final_conv = Conv["conv", spatial_dims](fea[5], out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor):
        # 编码器阶段
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        
        # 收集编码器特征
        encoder_features = [x0, x1, x2, x3, x4]
        
        # 解码器阶段
        u4 = self.upcat_4(x4, encoder_features)
        u3 = self.upcat_3(u4, encoder_features)
        u2 = self.upcat_2(u3, encoder_features)
        u1 = self.upcat_1(u2, encoder_features)
        
        logits = self.final_conv(u1)
        return logits


if __name__ == '__main__':
    from torchsummary import summary
    model = CrossBasicUNet(spatial_dims=2, in_channels=1, out_channels=1, features=[32, 32, 64, 128, 256, 32])
    summary(model, (1, 256, 256))

