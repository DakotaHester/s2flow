import torch
import torch.nn as nn

from typing import Literal


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