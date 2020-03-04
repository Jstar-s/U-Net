import torch.nn as nn
import torch
from torch.nn import functional as F


class Unet_down(nn.Module):
    def __init__(self,inchannel, outchannel):
        super(Unet_down,self).__init__()
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(outchannel, inchannel,kernel_size=3, stride=1, padding=0)
        self.pooling = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pooling(out)
        return out


class U_netup(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(U_netup, self).__init__()
        self.up1 = nn.Conv2d(in_channel, out_channel, kernel_size=2, padding=0)

    def forward(self, x):
        out = self.up1(x)
        return x


class U_net(nn.Module):
    def __init__(self):
        super(U_net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=(1, 1))
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=(1, 1))
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=(1, 1))
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=(1,1))
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=(1, 1))
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=(1, 1))
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=(1, 1))
        self.drop5 = nn.Dropout(0.5)

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=(2, 2))
        self.up6 = nn.Conv2d(1024, 512, kernel_size=3, padding=(1, 1))
        # 512+512 = 1024 drop4+up6
        self.conv11 = nn.Conv2d(1024, 512, kernel_size=3)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3)

        self.upsample2 = nn.UpsamplingBilinear2d(size=(65, 65))
        self.up7 = nn.Conv2d(512, 256, kernel_size=2, padding=0)
        # conv6 + up7 256+256 =512
        self.conv13 = nn.Conv2d(512, 256, kernel_size=3, padding=(1, 1))
        self.conv14 = nn.Conv2d(256, 256, kernel_size=3, padding=(1, 1))

        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=(2, 2))
        self.up8 = nn.Conv2d(256, 128, kernel_size=3, padding=(1, 1))
        # conv4 + up8 128+128 =256
        self.conv15 = nn.Conv2d(256, 128, kernel_size=3, padding=(1, 1))
        self.conv16 = nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1))

        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=(2, 2))
        self.up9 = nn.Conv2d(128, 64, kernel_size=3, padding=(1, 1))
        # conv2 + up9 64+64 =128
        self.conv17 = nn.Conv2d(128, 64, kernel_size=3, padding=(1, 1))
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1))

        self.conv19 = nn.Conv2d(64, 2, kernel_size=3, padding=(1, 1))
        self.conv20 = nn.Conv2d(2, 1, kernel_size=1)
    def forward(self, x):
        conv1 = F.relu(self.conv2((F.relu(self.conv1(x)))))
        pool1 = self.pool1(conv1)

        conv2 = F.relu(self.conv4((F.relu(self.conv3(pool1)))))
        pool2 = self.pool2(conv2)

        conv3 = F.relu(self.conv6((F.relu(self.conv5(pool2)))))
        pool3 = self.pool3(conv3)

        conv4 = F.relu(self.conv8((F.relu(self.conv7(pool3)))))
        drop4 = self.drop4(conv4)
        pool4 = self.pool3(drop4) # torch.Size([5, 512, 24, 24])

        conv5 = F.relu(self.conv10((F.relu(self.conv9(pool4)))))
        drop5 = self.drop5(conv5) #torch.Size([5, 1024, 8, 8])

        up6 = F.relu(self.up6(self.upsample1(drop5)))
        merge6 = torch.cat((drop4, up6), dim=1)
        conv6 = F.relu(self.conv12(F.relu(self.conv11(merge6))))

        upsample2 = self.upsample2(conv6)
        up7 = F.relu(self.up7(upsample2))
        merge7 = torch.cat((conv3, up7), dim=1)
        conv7 = F.relu(self.conv14(F.relu(self.conv13(merge7))))

        up8 = F.relu(self.up8(self.upsample1(conv7)))
        merge8 = torch.cat((conv2, up8), dim=1)
        conv8 = F.relu(self.conv16(F.relu(self.conv15(merge8))))

        up9 = F.relu(self.up9(self.upsample1(conv8)))
        merge9 = torch.cat((conv1, up9), dim=1)
        conv9 = F.relu(self.conv18(F.relu(self.conv17(merge9))))
        conv9 = F.relu(self.conv19(conv9))
        conv10 = torch.sigmoid(self.conv20(conv9))
        return conv10


def main():
    x = torch.randn(1, 1, 256, 256)
    print(x.shape)
    model = U_net()
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()