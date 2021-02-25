import paddle
import paddle.nn as nn

class Unet(nn.Layer):
    def __init__(self, num_classes=19):
        super(Unet, self).__init__()

        channels = 64
        filters = [channels, channels*2, channels*4, channels*8, channels*16]

        self.downconv1 = DoubleConv(3, filters[0])

        self.down1 = DownSample(filters[0], filters[1])
        self.down2 = DownSample(filters[1], filters[2])
        self.down3 = DownSample(filters[2], filters[3])
        self.down4 = DownSample(filters[3], filters[4])

        self.up4 = UpSample(filters[4], filters[3])
        self.upconv4 = DoubleConv(filters[4], filters[3])
        self.up3 = UpSample(filters[3], filters[2])
        self.upconv3 = DoubleConv(filters[3], filters[2])
        self.up2 = UpSample(filters[2], filters[1])
        self.upconv2 = DoubleConv(filters[2], filters[1])
        self.up1 = UpSample(filters[1], filters[0])
        self.upconv1 = DoubleConv(filters[1], filters[0])

        self.out = nn.Conv2D(filters[0], num_classes, 1)

    def forward(self, x):
        c1 = self.downconv1(x)
        d1 = self.down1(c1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u4 = self.up4(d4)
        u3 = self.up3(self.upconv4(paddle.concat((d3, u4), axis=1)))
        u2 = self.up2(self.upconv3(paddle.concat((d2, u3), axis=1)))
        u1 = self.up1(self.upconv2(paddle.concat((d1, u2), axis=1)))

        output = self.out(self.upconv1(paddle.concat((c1, u1), axis=1)))
        output = output.transpose([0, 2, 3, 1])

        return output

class DoubleConv(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU(),
            nn.Conv2D(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2D(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_sample(x)


class UpSample(nn.Layer):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.up_sample(x)
