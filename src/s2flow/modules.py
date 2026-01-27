import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

from typing import Any, Dict, Literal


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ConvBlock(nn.Module):
    
    """A simple convolutional block followed by batch normalization and ReLU activation."""
    def __init__(self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int=3, 
        stride: int=1, 
        padding: int=1, 
        batch_norm: int=True, 
        activation: Literal['relu', 'leaky_relu', 'sigmoid', 'softmax', 'tanh', 'swish']='relu',
    ):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm = batch_norm
        self.activation = activation
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
            
        if activation is not None:
            if activation == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif activation == 'gelu':
                self.act = nn.GELU()
            elif activation == 'leaky_relu':
                self.act = nn.LeakyReLU(inplace=True)
            elif activation == 'sigmoid':
                self.act = nn.Sigmoid()
            elif activation == 'softmax':
                self.act = nn.Softmax(dim=1)
            elif activation == 'tanh':
                self.act = nn.Tanh()
            elif activation == 'swish':
                self.act = nn.SiLU()
            else:
                raise ValueError(f'Invalid value for `activation`: {activation}. Supported values are ["relu", "leaky_relu", "sigmoid", "softmax", "tanh"].')
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        
        if self.activation is not None:
            x = self.act(x)
        
        if self.enable_cbam:
            x = self.cbam(x)
        return x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling as in DeepLab v3+."""
    def __init__(self, in_channels: int, out_channels: int, dilation_rates: tuple[int, ...]) -> None:
        super().__init__()
        # 1×1 conv branch
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # parallel atrous conv branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            for rate in dilation_rates
        ])
        # image-level pooling branch
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # combine & project
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (2 + len(dilation_rates)), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        feats = [self.conv_1x1(x)] + [branch(x) for branch in self.branches]
        # image-level features
        img_feat = self.image_pool(x)
        img_feat = nn.functional.interpolate(img_feat, size=size, mode="bilinear", align_corners=False)
        feats.append(img_feat)
        x = torch.cat(feats, dim=1)
        return self.project(x)


class Decoder(nn.Module):
    """DeepLab v3+ decoder that fuses low- and high-level features."""
    def __init__(self, low_level_in: int, low_level_out: int, num_classes: int) -> None:
        super().__init__()
        # Reduce low-level feature channels to low_level_out (e.g. 48)
        self.reduce_low = ConvBlock(low_level_in, low_level_out, kernel_size=1, padding=0, batch_norm=True, activation='relu')
        # Two separable conv layers to refine concatenated features
        self.refine = nn.Sequential(
            nn.DepthwiseSeparableConv(low_level_out + 256, 256, kernel_size=3, padding=1),
            nn.DepthwiseSeparableConv(256, 256, kernel_size=3, padding=1),
        )
        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, low_level_feat: torch.Tensor, high_level_feat: torch.Tensor) -> torch.Tensor:
        # Upsample ASPP output by factor 4
        high = nn.functional.interpolate(high_level_feat, size=low_level_feat.shape[-2:], mode="bilinear", align_corners=False)
        low = self.reduce_low(low_level_feat)
        x = torch.cat([low, high], dim=1)
        x = self.refine(x)
        return self.classifier(x)


class DeepLabV3Plus(nn.Module):
    """
    DeepLab v3+ for semantic segmentation.
    - backbone: module returning (low_level_feat, high_level_feat)
    - num_classes: # of segmentation classes
    - aspp_rates: dilation rates for ASPP
    """
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        aspp_out: int = 256,
        aspp_rates: tuple[int, ...] = (12, 24, 36),
    ) -> None:
        super().__init__()
        self.backbone = backbone
        # ASPP on high-level features
        self.aspp = ASPP(in_channels=2048, out_channels=aspp_out, dilation_rates=aspp_rates)
        # Decoder fusing ASPP and low-level (conv2) features
        self.decoder = Decoder(low_level_in=256, low_level_out=48, num_classes=num_classes)
        
        if num_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, low_level, _, _, high_level = self.backbone(x)
        x = self.aspp(high_level)
        x = self.decoder(low_level, x)
        # Final upsample to input resolution
        x = nn.functional.interpolate(x, size=x.shape[-2]*4, mode="bilinear", align_corners=False)
        return self.activation(x)



class RRDBNet(nn.Module):
    """
    The Generator for Real-ESRGAN.
    Structure: Conv -> RRDB Body -> Conv -> Upsample (Nearest+Conv) -> Conv -> Output
    """
    def __init__(self, model_conf: Dict[str, Any]):
        super(RRDBNet, self).__init__()
        in_channels = model_conf.get('in_channels', 4) 
        out_channels = model_conf.get('out_channels', 4)
        num_feat = model_conf.get('num_feat', 64)
        num_block = model_conf.get('num_block', 23)
        num_growth = model_conf.get('num_growth', 32)
        scale = model_conf.get('scale', 4)

        # 1. First convolution
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1)

        # 2. Main Body (RRDB blocks)
        self.body = make_layer(lambda: RRDB(num_feat, num_growth), num_block)
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # 3. Upsampling
        # Real-ESRGAN/ESRGAN uses Nearest Neighbor + Conv, NOT PixelShuffle
        upsample_layers = []
        num_upsamples = int(log2(scale))
        for _ in range(num_upsamples):
            upsample_layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]
        self.upsample = nn.Sequential(*upsample_layers)
        
        # 4. Final convolution
        self.conv_last_1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last_2 = nn.Conv2d(num_feat, out_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.upsample(feat)
        feat = self.conv_last_1(feat)
        feat = self.lrelu(feat)
        out = self.conv_last_2(feat)
        
        return out


class UNetDiscriminatorSN(nn.Module):
    """
    U-Net Discriminator with Spectral Normalization.
    Specific to Real-ESRGAN to provide pixel-wise loss gradients.
    """
    def __init__(self, config):
        super(UNetDiscriminatorSN, self).__init__()
        disc_conf = config.get('discriminator_model', {})
        in_channels = disc_conf.get('in_channels', 4)
        num_feat = disc_conf.get('num_feat', 64)
        self.skip_connection = disc_conf.get('skip_connection', True)
        
        norm = nn.utils.spectral_norm

        self.conv0 = norm(nn.Conv2d(in_channels, num_feat, 3, 1, 1))

        # Downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))

        # Upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))

        # Output
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)

        # Down
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # Up (Using Interpolation + Conv for stability)
        x3_up = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3_up), negative_slope=0.2, inplace=True)
        if self.skip_connection: x4 = x4 + x2

        x4_up = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4_up), negative_slope=0.2, inplace=True)
        if self.skip_connection: x5 = x5 + x1

        x5_up = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5_up), negative_slope=0.2, inplace=True)
        if self.skip_connection: x6 = x6 + x0

        # Refinement & Output
        out = self.conv7(x6)
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)
        out = self.conv8(out)
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)
        out = self.conv9(out)
        
        return out

class ResidualDenseBlock_RRDB(nn.Module):
    """
    Residual Dense Block (RDB) for RRDB.
    Standard Real-ESRGAN/ESRGAN configuration.
    """
    def __init__(self, num_feat=64, num_growth=32):
        super(ResidualDenseBlock_RRDB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_growth, num_growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_growth, num_growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_growth, num_growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_growth, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Initialization (Kaiming)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Residual scaling 0.2 (Critical for convergence in deep networks)
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB).
    """
    def __init__(self, num_feat, num_growth=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_RRDB(num_feat, num_growth)
        self.rdb2 = ResidualDenseBlock_RRDB(num_feat, num_growth)
        self.rdb3 = ResidualDenseBlock_RRDB(num_feat, num_growth)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Residual scaling 0.2
        return out * 0.2 + x

