import os
import numpy as np
import PIL.Image
import torch
from torch.utils import data
import torchvision.transforms
import random


class VGG_Faces(data.Dataset):
    mean_bgr = np.array([93.5940, 104.7624, 129.1863])  # from resnet50_ft.prototxt

    def __init__(self, root, image_list_file, split='train_occ_7',
                 o_width=0, o_height=0, r_start=0, c_start=0, transform=True,
                 upper=None):
        """
        :param root: dataset directory
        :param image_list_file: contains image file names under root
        :param image_list_file: contains image file names under root
        :param id_label_dict: X[class_id] -> label
        :param split: train or valid
        :param transform:
        :param upper: max number of image used for debug
        """
        assert os.path.exists(root), "root: {} not found.".format(root)
        # 'E:/Coding/data/testdata'
        self.root = root
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self.split = split
        self._transform = transform
        self.o_width = o_width
        self.o_height = o_height
        self.r_start = r_start
        self.c_start = c_start

        self.img_info = []
        with open(self.image_list_file, 'r') as f:
            for i, img_file in enumerate(f):
                img_file = img_file.strip()  # e.g. train/n004332/0317_01.jpg
                class_name = img_file.split("/")[1]  # like n004332
                self.img_info.append({
                    'cname': class_name,
                    'img': img_file
                })
                if i % 1000 == 0:
                    print("processing: {} images for {}".format(i, self.split))
                if upper and i == upper - 1:  # for debug purpose
                    break

        #np.random.seed(666)
        self.occ_idxs = np.random.randint(1, 21, size=len(self.img_info))
        print(len(self.occ_idxs))

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]
        occ_id = self.occ_idxs[index]
        img_file = info['img']  # e.g. train/A.J._Buckley/00000001.jpg 
        img = PIL.Image.open(os.path.join(self.root, img_file))
        img = self.process_image_channels(img, os.path.join(self.root, img_file))

        img = np.array(img, dtype=np.uint8)
        assert len(img.shape) == 3  # assumes color images and no alpha channel
        # generate occ image
        img_occ = self.generate_occ_img(img, occ_id)
        self.save_occ_img(img_occ, occ_id, img_file)
        class_name = info['cname']
        if self._transform:
            return self.transform(img), self.transform(img_occ), img_file, class_name
        else:
            return img, img_occ, img_file, class_name

    def generate_occ_img(self, img, occ_id):
        occ_img_path = self.root + '/occ_pattern/occ' + str(occ_id) + '.jpg'
        occ_img = PIL.Image.open(occ_img_path)
        occ_img = torchvision.transforms.Resize((self.o_height, self.o_width))(occ_img)
        occ_img = np.array(occ_img, dtype=np.uint8)
        img_occ = np.copy(img)
        img_occ[self.r_start:self.r_start + self.o_height, self.c_start:self.c_start + self.o_width, :] = occ_img[:, :, :]
        return img_occ

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img

    def save_occ_img(self, img_occ, occ_id, img_file):
        # train/A.J._Buckley/00000001.jpg
        class_name = img_file.split("/")[1]
        image_name = img_file.split("/")[2]
        image_id = image_name.split(".")[0]
        save_folder = self.root + '/' + self.split + '/' + class_name
        '''
        if False == os.path.exists(save_folder):
            
            os.makedirs(save_folder)
        '''
        save_path = save_folder + '/' + image_id + '_' + str(occ_id) + '.jpg'
        PIL.Image.fromarray(img_occ).save(save_path)

    def process_image_channels(self, img, image_path):
        # process the 4 channels .png
        if img.mode == 'RGBA':
            r, g, b, a = img.split()
            img = PIL.Image.merge("RGB", (r, g, b))
        # process the 1 channel image
        elif img.mode != 'RGB':
            img = img.convert("RGB")
            os.remove(image_path)
            img.save(image_path)
        return img

    '''
    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        return img, lbl
    '''
