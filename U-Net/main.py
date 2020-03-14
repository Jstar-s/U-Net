import  torch
from    torch import optim, nn
import  torchvision
from    torch.utils.data import DataLoader
# from    resnet import ResNet18
from    torchvision.models import resnet18
from data_utls import Unet
from torchvision.transforms import ToPILImage
from  Unet import U_net
import numpy as np
from PIL import Image


batchsz = 32
lr = 1e-4
epochs = 20

device = torch.device('cuda')
torch.manual_seed(1234)
train_db = Unet(".\\membrane\\train", 256, "train")
train_loader = DataLoader(train_db, batch_size=1, shuffle=True, num_workers=1)


def evalute(model, loader):
    pass


def main():
    model = U_net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.BCELoss()
    best_acc, best_epoch = 0, 0
    global_step = 0

    for epoch in range(epochs):

        for step, (x, y) in enumerate(train_loader):

            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            B, C, H, W = logits.shape
            y = np.squeeze(y)
            logits = np.squeeze(logits)
            logits[logits>0.5] = 1
            logits[logits <= 0.5] = 0
            acc = torch.mean((y == logits), dtype=float)
            print(acc)
            # val = logits.cuda().data.cpu().numpy()
            # val = np.squeeze(val)
            # val = Image.fromarray(val)
            # val.show()


if __name__ == "__main__":
    main()