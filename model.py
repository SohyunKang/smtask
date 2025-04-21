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