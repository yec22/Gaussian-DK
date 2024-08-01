import torch
import torch.nn as nn

class LightnessMapper(nn.Module):
    def __init__(self, in_channel, hidden_channel=16):
        super(LightnessMapper, self).__init__()
        self.net = nn.Sequential(
            # input: [ic, H, W]
            nn.Conv2d(in_channel, hidden_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # [hc, H, W]
            nn.Conv2d(hidden_channel, hidden_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # [hc, H, W]
            nn.Conv2d(hidden_channel, 1, 3, 1, 1, bias=True)
            # output: [1, H, W]
        )

    def forward(self, EV, light_feature):
        input = torch.cat([EV, light_feature], dim=0)
        output =  self.net(input) + EV
        return output
    
class ToneMapper(nn.Module):
    def __init__(self, in_channel=1, hidden_channel=16):
        super(ToneMapper, self).__init__()
        self.R_layer = nn.Sequential(
            # input: [ic, H, W]
            nn.Conv2d(in_channel, hidden_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # [hc, H, W]
            nn.Conv2d(hidden_channel, hidden_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # [hc, H, W]
            nn.Conv2d(hidden_channel, 1, 3, 1, 1, bias=True),
            nn.Sigmoid()
            # output: [1, H, W]
        )
        self.G_layer = nn.Sequential(
            # input: [ic, H, W]
            nn.Conv2d(in_channel, hidden_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # [hc, H, W]
            nn.Conv2d(hidden_channel, hidden_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # [hc, H, W]
            nn.Conv2d(hidden_channel, 1, 3, 1, 1, bias=True),
            nn.Sigmoid()
            # output: [1, H, W]
        )
        self.B_layer = nn.Sequential(
            # input: [ic, H, W]
            nn.Conv2d(in_channel, hidden_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # [hc, H, W]
            nn.Conv2d(hidden_channel, hidden_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # [hc, H, W]
            nn.Conv2d(hidden_channel, 1, 3, 1, 1, bias=True),
            nn.Sigmoid()
            # output: [1, H, W]
        )

    def forward(self, HDR):
        HDR_R, HDR_G, HDR_B = HDR[0:1, ...], HDR[1:2, ...], HDR[2:3, ...]
        LDR_R = self.R_layer(HDR_R)
        LDR_G = self.G_layer(HDR_G)
        LDR_B = self.B_layer(HDR_B)
        output = torch.cat([LDR_R, LDR_G, LDR_B], dim=0)
        return output
    
class ToneMapper_RGB_combine(nn.Module):
    def __init__(self, in_channel=3, hidden_channel=16):
        super(ToneMapper_RGB_combine, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channel, in_channel, kernel_size=3, padding=1)
        )

    def forward(self, input):
        return self.net(input)