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

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv9 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0)
        self.conv10 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)
        self.drop4 = nn.Dropout(0.5)
        self.pool5 = nn.MaxPool2d(2, stride=2)

        self.up6 = nn.Conv2d(1024, 512, kernel_size=3)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3)
        self.drop1 = nn.Dropout(0.5)
        self.pooling1 = nn.MaxPool2d(2,stride=2)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=3)
        self.drop = nn.Dropout(0.5)

        self.up1 = nn.Conv2d(1024, 512, kernel_size=2)


def main():
    x = torch.randn(5, 3, 128, 128)
    print(x.shape)
    model = Unet_down(3, 64)
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()