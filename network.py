# 计算可训练参数

import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

class MappingNetwork(nn.Module):
    def __init__(self, features: int, n_layers: int):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(features, features))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, dim=1)  # 归一化
        return self.net(z)


class Generator(nn.Module):
    """3D生成器：从4×4×4常量体积逐步上采样到64×64×64"""

    def __init__(self, log_resolution: int, d_latent: int, n_features: int = 8, max_features: int = 128):
        super().__init__()
        # log_resolution-2 = 6-2 = 4     i range( 4,3,2,1,0)
        # 计算各块的特征数  min( 128 , 32*(2 ** i) )  =>  [128，64，32，16，8]
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)  # 5

        # 3D初始常量 (1, 128, 4, 4, 4)
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4, 4)))

        # 第一个风格块（4×4×4）和3D体积转换层
        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_volume = ToVolume(d_latent, features[0])  # 替换ToRGB为ToVolume（单通道3D）

        # 生成器块（逐步上采样到64×64×64）
        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

        # 3D上采样层
        self.up_sample = UpSample3D()

        self.activation = nn.Tanh()

    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        batch_size = w.shape[1]
        # 扩展初始常量到批次大小（3D）  (3, 128, 4, 4, 4)
        x = self.initial_constant.expand(batch_size, -1, -1, -1, -1)

        # 第一个风格块
        x = self.style_block(x, w[0], input_noise[0][1])
        # 初始3D体积
        volume = self.to_volume(x, w[0])

        # 逐步上采样到目标分辨率
        for i in range(1, self.n_blocks):
            x = self.up_sample(x)  # 3D上采样
            x, volume_new = self.blocks[i - 1](x, w[i], input_noise[i])
            volume = self.up_sample(volume) + volume_new  # 累加各层体积

        return self.activation(volume)


class GeneratorBlock(nn.Module):
    """3D生成器块：包含两个3D风格块和3D体积输出"""

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)
        self.to_volume = ToVolume(d_latent, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        x = self.style_block1(x, w, noise[0])  # 3D风格块1
        x = self.style_block2(x, w, noise[1])  # 3D风格块2
        volume = self.to_volume(x, w)  # 3D体积输出
        return x, volume


class StyleBlock(nn.Module):
    """3D风格块：权重调制3D卷积"""

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv3dWeightModulate(in_features, out_features, kernel_size=3)  # 3D卷积
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        s = self.to_style(w)  # 风格向量
        x = self.conv(x, s)  # 3D权重调制卷积
        # 添加3D噪声
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None, None])


class ToVolume(nn.Module):
    """3D体积转换层：从特征图生成单通道3D体积"""

    def __init__(self, d_latent: int, features: int):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)
        self.conv = Conv3dWeightModulate(features, 1, kernel_size=1, demodulate=False)  # 输出单通道
        self.bias = nn.Parameter(torch.zeros(1))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        style = self.to_style(w)
        x = self.conv(x, style)  # 3D卷积
        return self.activation(x + self.bias[None, :, None, None, None])


class Conv3dWeightModulate(nn.Module):
    """3D权重调制与去调制卷积"""

    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: bool = True, eps: float = 1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2  # 3D padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size, kernel_size])  # 3D权重
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        b, _, d, h, w = x.shape  # 3D维度：depth, height, width
        s = s[:, None, :, None, None, None]  # 适配3D权重形状
        weights = self.weight()[None, :, :, :, :, :]
        weights = weights * s  # 权重调制

        # 去调制（3D）
        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        # 3D分组卷积
        x = x.reshape(1, -1, d, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)
        x = F.conv3d(x, weights, padding=self.padding, groups=b)  # 3D卷积
        return x.reshape(-1, self.out_features, d, h, w)


class Discriminator(nn.Module):
    """3D判别器：处理128×128×128单通道体积"""

    def __init__(self, log_resolution: int, n_features: int = 8, max_features: int = 128):
        super().__init__()
        # 从单通道3D体积转换为特征图
        self.from_volume = nn.Sequential(
            EqualizedConv3d(1, n_features, 1),  # 3D卷积
            nn.LeakyReLU(0.2, True),
        )

        # 特征数计算（3D）
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        n_blocks = len(features) - 1
        self.blocks = nn.Sequential(*[DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)])

        # 3D迷你批次标准差 (3, 129, 4, 4, 4)
        self.std_dev = MiniBatchStdDev3D()
        final_features = features[-1] + 1  # 129
        # # 3D卷积   新添加padding=1确保形状不变
        self.conv = EqualizedConv3d(final_features, final_features, 3, padding=1)
        # 最终特征图尺寸为 4×4×4
        self.final = EqualizedLinear(4 * 4 * 4 * final_features, 1)  # 适配3D最终尺寸

    def forward(self, x: torch.Tensor):
        x = self.from_volume(x)  # 从3D体积到特征图
        x = self.blocks(x)
        x = self.std_dev(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)  # 展平3D特征 （3，513*4*4*4）
        return self.final(x)  # （3，1）


class DiscriminatorBlock(nn.Module):
    """3D判别器块：3D卷积+残差连接"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(
            DownSample3D(),  # 3D下采样
            EqualizedConv3d(in_features, out_features, kernel_size=1)  # 3D卷积
        )
        self.block = nn.Sequential(
            EqualizedConv3d(in_features, in_features, kernel_size=3, padding=1),  # 3D卷积
            nn.LeakyReLU(0.2, True),
            EqualizedConv3d(in_features, out_features, kernel_size=3, padding=1),  # 3D卷积下采样
            nn.LeakyReLU(0.2, True),
        )
        self.down_sample = DownSample3D()
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        # 残差路径：下采样+1×1卷积
        residual = self.residual(x)
        # 主路径：卷积+下采样
        x = self.block(x)
        x = self.down_sample(x)
        return (x + residual) * self.scale  # # 融合残差与主路径


class MiniBatchStdDev3D(nn.Module):
    """3D迷你批次标准差：计算3D特征图批次内标准差"""

    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    ####   参考ProGAN源代码修改，保留D，H，W维度
    def forward(self, x: torch.Tensor):
        # x形状：[N, C, D, H, W]
        G = self.group_size
        N, C, D, H, W = x.shape
        assert N % G == 0, "批量大小必须是group_size的整数倍"
        M = N // G  # 组数

        # 分组为 [G, M, C, D, H, W]（保留3D空间维度）
        grouped = x.view(G, M, C, D, H, W)
        # 计算组内均值（沿G维度）
        mean = grouped.mean(dim=0, keepdim=True)  # [1, M, C, D, H, W]
        # 中心化
        grouped = grouped - mean
        # 计算组内方差（沿G维度）
        var = grouped.var(dim=0, unbiased=False)  # [M, C, D, H, W]
        # 标准差
        std = torch.sqrt(var + 1e-8)  # [M, C, D, H, W]
        # 沿通道、深度、高度、宽度取平均，得到每个组的平均标准差
        std = std.mean(dim=[1, 2, 3, 4], keepdim=True)  # [M, 1, 1, 1, 1]
        # 扩展到匹配原批量大小和空间维度
        std = std.repeat_interleave(G, dim=0)  # [N, 1, 1, 1, 1]
        std = std.expand(-1, -1, D, H, W)  # [N, 1, D, H, W]
        # 拼接
        return torch.cat([x, std], dim=1)  # [N, C+1, D, H, W]


class DownSample3D(nn.Module):
    """3D下采样：平滑+3D下采样"""

    def __init__(self):
        super().__init__()
        self.smooth = Smooth3D()

    def forward(self, x: torch.Tensor):
        x = self.smooth(x)
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2, x.shape[4] // 2),
                             mode='trilinear', align_corners=False)  # 3D插值


class UpSample3D(nn.Module):
    """3D上采样：3D上采样+平滑"""

    def __init__(self):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)  # 3D上采样
        self.smooth = Smooth3D()

    def forward(self, x: torch.Tensor):
        return self.smooth(self.up_sample(x))


class Smooth3D(nn.Module):
    """3D平滑层：3D高斯模糊核"""

    def __init__(self):
        super().__init__()
        # 3D平滑核（1x3x3x3x1）
        kernel = torch.tensor([[[[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                                 [[2, 4, 2], [4, 8, 4], [2, 4, 2]],
                                 [[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReplicationPad3d(1)  # 3D padding

    def forward(self, x: torch.Tensor):
        b, c, d, h, w = x.shape
        x = x.view(-1, 1, d, h, w)
        x = self.pad(x)
        x = F.conv3d(x, self.kernel)  # 3D卷积平滑
        return x.view(b, c, d, h, w)


# 以下类保持不变，但适配3D输入（无需修改核心逻辑）
class EqualizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv3d(nn.Module):
    """3D均等学习率卷积"""

    def __init__(self, in_features: int, out_features: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size, kernel_size])  # 3D权重
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv3d(x, self.weight(), bias=self.bias, padding=self.padding)  # 3D卷积


class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c



