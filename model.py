import torch
import torch.nn as nn

# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# Conv Block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# Attention U-Net
class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features = [32, 64, 128, 256]):
        super(AttentionUNet, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2)

        prev_ch = in_ch
        for feat in features:
            self.encoder_blocks.append(ConvBlock(prev_ch, feat))
            prev_ch = feat

        # self.bottleneck = ConvBlock(prev_ch, prev_ch * 2)
        self.bottleneck = ConvBlock(prev_ch, prev_ch)

        self.att_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(prev_ch, feat, kernel_size=2, stride=2))  # ✅ 수정됨
            self.att_blocks.append(AttentionBlock(F_g=feat, F_l=feat, F_int=feat // 2))
            self.decoder_blocks.append(ConvBlock(feat * 2, feat))
            prev_ch = feat

        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        enc_features = []

        for block in self.encoder_blocks:
            x = block(x)
            enc_features.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = enc_features[-(i + 1)]
            attn = self.att_blocks[i](g=x, x=skip)
            x = torch.cat((attn, x), dim=1)
            x = self.decoder_blocks[i](x)

        return self.final_conv(x)
    
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# 기존 AttentionBlock, ConvBlock은 그대로 유지

class AttentionUNetResNet18(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, pretrained=True):
        super().__init__()
        
        # ✅ 최신 권장 방식으로 pretrained weight 불러오기
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        if in_ch != 3:
            self.input_conv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            resnet.conv1 = self.input_conv

        # Encoder (ResNet)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # [B, 64, H/2]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                   # [B, 64, H/4]
        self.encoder2 = resnet.layer1  # [B, 64, H/4]
        self.encoder3 = resnet.layer2  # [B, 128, H/8]
        self.encoder4 = resnet.layer3  # [B, 256, H/16]
        self.bottleneck = resnet.layer4  # [B, 512, H/32]

        # Decoder + Attention
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(256, 256, 128)
        self.dec4 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(128, 128, 64)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(64, 64, 32)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(64, 64, 32)
        self.dec1 = ConvBlock(128, 64)

        # ✅ 추가 업샘플링 (H/64 → H/32 → H/16 → H/8 → H/4 → H/2 → H)
        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(64, 64)

        self.final_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)    # H/2
        e1p = self.pool1(e1)     # H/4
        e2 = self.encoder2(e1p)  # H/4
        e3 = self.encoder3(e2)   # H/8
        e4 = self.encoder4(e3)   # H/16
        x = self.bottleneck(e4)  # H/32

        # Decoder
        x = self.up4(x)
        x = self.dec4(torch.cat([self.att4(x, e4), x], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([self.att3(x, e3), x], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([self.att2(x, e2), x], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([self.att1(x, e1), x], dim=1))

        # ✅ 추가 업샘플링
        x = self.up0(x)
        x = self.dec0(x)

        return self.final_conv(x)
