import  torch
from    torch import optim, nn
import  torchvision
from    torch.utils.data import DataLoader
# from    resnet import ResNet18
from    torchvision.models import resnet18
from data_utls import Unet
from torchvision.transforms import ToPILImage
from  Unet import U_net

batchsz = 32
lr = 1e-3
epochs = 10

device = torch.device('cuda')
torch.manual_seed(1234)

train_db = Unet(".\\membrane\\train", 256, "train")
train_loader = DataLoader(train_db, batch_size=2, shuffle=True, num_workers=2)


def evalute(model, loader):
    pass


def main():
    model = U_net()


if __name__ == "__main__":
    main()