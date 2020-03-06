import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import  os, glob
import  random, csv

class Unet(data.Dataset):
    # path对应图片路径， resize指定输出的图片大小，mode指定训练、测试等模式
    def __init__(self, path, resize, mode):
        super(Unet, self).__init__()

        self.path = path
        self.resize = resize
        self.mode = mode
        self.image_dir = os.listdir(os.path.join(path, 'images'))
        self.label_dir = os.listdir(os.path.join(path, 'labels'))
        if self.mode == 'train':
            self.images = [os.path.join(path, 'images', img) for img in self.image_dir]
            print(self.images)
            self.labels = [os.path.join(path, 'labels', img) for img in self.label_dir]
            print(self.labels)
        if self.mode == 'test':
            self.images = [os.path.join(path, 'images', img) for img in self.image_dir]
            print(self.images)
            self.labels = [os.path.join(path, 'labels', img) for img in self.label_dir]
            print(self.labels)

    def __getitem__(self, index):
        images = Image.open(self.images[index])
        labels = Image.open(self.labels[index])
        tf = tfs.Compose([
            tfs.RandomRotation(15),
            tfs.Resize((self.resize, self.resize)),
            tfs.CenterCrop(self.resize),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0], std=[1])
        ])
        # images = tfs.ToTensor()(images)
        # images=tfs.Normalize(mean=[0], std=[1])(images)
        # labels = tfs.ToTensor()(labels)
        # labels = tfs.Normalize(mean=[0], std=[1])(labels)
        images = tf(images)
        labels = tf(labels)
        return images, labels

    def __len__(self):
        return len(self.images)


def main():

    data = Unet(".\\membrane\\train", 256, "train")
    train_loader = DataLoader(dataset=Unet(".\\membrane\\train", 256, "train"))
    x, y = next(iter(data))

    x = ToPILImage()(x)  # tensor转为PIL Image
    y = ToPILImage()(y)  # tensor转为PIL Image

    x.show()  # 显示图片
    y.show()
    print(x.size)


if __name__ == "__main__":
    main()