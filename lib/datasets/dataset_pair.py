import torch.utils.data as data
from PIL import Image, ImageFile
import os

ImageFile.LOAD_TRUNCATED_IAMGES = True


# https://github.com/pytorch/vision/issues/81

def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)


def default_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, imgOccPath, label = line.strip().split('\t')
            imgPair = imgPath + '\t' + imgOccPath
            imgList.append((imgPair, int(label)))
    return imgList


class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        imgPair, target = self.imgList[index]
        imgPath, imgOccPath = imgPair.split('\t')
        img = self.loader(os.path.join(self.root, imgPath))
        img_occ = self.loader(os.path.join(self.root, imgOccPath))

        if self.transform is not None:
            img = self.transform(img)
            img_occ = self.transform(img_occ)
        return img, img_occ, imgPair, target

    def __len__(self):
        return len(self.imgList)
