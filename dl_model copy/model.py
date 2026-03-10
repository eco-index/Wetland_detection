import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8, dropout=0.0):
        super().__init__()
        g = min(groups, out_ch) if out_ch % groups == 0 else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g, out_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.drop1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g, out_ch)
        self.act2 = nn.SiLU(inplace=True)
        self.drop2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = self.act2(x)
        x = self.drop2(x)

        return x


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Match size if needed
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        x = x_flat + self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))[0]
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x


class TransformerUNet(nn.Module):
    def __init__(self, in_channels=12, num_classes=1, base_channels=64):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = DownBlock(base_channels, base_channels * 2)
        self.enc3 = DownBlock(base_channels * 2, base_channels * 4)
        self.enc4 = DownBlock(base_channels * 4, base_channels * 8)

        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 8, base_channels * 8),
            TransformerBlock(base_channels * 8),
        )

        self.dec4 = UpBlock(base_channels * 8 + base_channels * 8, base_channels * 4)
        self.dec3 = UpBlock(base_channels * 4 + base_channels * 4, base_channels * 2)
        self.dec2 = UpBlock(base_channels * 2 + base_channels * 2, base_channels)
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec1 = nn.Sequential(
            ConvBlock(base_channels + base_channels, base_channels),
            nn.Conv2d(base_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        x5 = self.bottleneck(x4)

        x = self.dec4(x5, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.final_up(x)

        # ✅ Fix shape mismatch before concat
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=True)

        x = self.dec1(torch.cat([x, x1], dim=1))

        return x
