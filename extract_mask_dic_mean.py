from __future__ import print_function
from __future__ import division
import argparse
import os
import time
import gc


import torch
import torchvision
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
cudnn.benchmark = True

import _init_paths
import models.net_sia as net_sia
import datasets.dataset_pair as dset
import lfw_eval
import layer
import utils

plt.switch_backend('agg')
parser = argparse.ArgumentParser(description='PyTorch CosFace')
# DATA
parser.add_argument('--root_path', type=str, default='/home/lingxuesong/data/sia',
                    help='path to root path of images')
parser.add_argument('--train_list', type=str, default=None,help='path to training list')
parser.add_argument('--batch_size', type=int, default=512,
                    help='input batch size for training (default: 512)')
parser.add_argument('--is_gray', type=bool, default=False,
                    help='Transform input image to gray or not  (default: False)')
# Network
parser.add_argument('--weight_model', type=str, default='')
parser.add_argument('--save_path', type=str, default='masks/mean/d12/',
                    help='path to save checkpoint')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='disables CUDA training')
parser.add_argument('--workers', type=int, default=8,
                    help='how many workers to load data')
parser.add_argument('--gpus', type=str, default='4,5,6,7')
parser.add_argument('--ngpus', type=int, default=4)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


def main():
    # --------------------------------------model----------------------------------------
    model = net_sia.LResNet50E_IR_Sia(is_gray=args.is_gray)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    model = model.to(device)
    
    args.run_name = utils.get_run_name()  # Dec25-14-53-43_lingxuesong-PC0
    output_dir = os.path.join(args.save_path, args.network)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # ------------------------------------load image---------------------------------------
    if args.is_gray:
        train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])  # gray
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    train_loader = torch.utils.data.DataLoader(
        dset.ImageList(root=args.root_path, fileList=args.train_list,
                  transform=train_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    print('length of train Database: ' + str(len(train_loader.dataset)) + ' Batches: ' + str(len(train_loader)))
    # ----------------------------------------train----------------------------------------
    extract(train_loader, model, args.weight_model, output_dir)
    print('Finished Extracting')
    

def extract(val_loader, model, model_path, output_dir):
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    OUT_SUM = np.zeros([512*7*6])
    count_p = 0
    with torch.no_grad():
        for batch_idx, (data, data_occ, pair_files, _) in enumerate(val_loader, 1):
            
            
            data, data_occ = data.to(device), data_occ.to(device)
            # compute output
            _, _, _, _, _, out = model(data, data_occ)
            
            print('Extracting for No.:{} batch...shape is:{}'.format(batch_idx + 1, out.shape))

            assert out.shape[0] == len(pair_files)

            for i, pair_file in enumerate(pair_files):
                out_cpu = out[i].to('cpu')
                out_cpu = out_cpu.view(-1,1)
                min_max_scaler = sklearn.preprocessing.MinMaxScaler()
                out_cpu = min_max_scaler.fit_transform(out_cpu)
                OUT_SUM = OUT_SUM + out_cpu.flatten()
                count_p = count_p + 1
                
    
    OUT_MEAN = OUT_SUM / count_p
    
    feature_file = os.path.join(output_dir, 'mean_mask.txt')
    np.savetxt(feature_file, OUT_MEAN)



if __name__ == '__main__': 
    print(args)
    main()
