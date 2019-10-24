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
import matplotlib.pyplot as plt
cudnn.benchmark = True

import _init_paths
import models.net_sia as net_sia
import datasets.dataset_pair as dset
import layer
import utils

plt.switch_backend('agg')
configurations = {
    1: dict(
        lr=1.0e-1,
        step_size=[7500, 15000, 22500],  # "lr_policy: step" [15, 30, 45, 50]
        epochs=50,
    ),
    2: dict(
        lr=1.0e-1,
        step_size=[10000, 17500, 22500],  # [20, 35, 45, 50] MultiStepLR
        epochs=50, 
    )
}


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CosFace')

# DATA
parser.add_argument('--root_path', type=str, default='/home/lingxuesong/data/sia',
                    help='path to root path of images')
parser.add_argument('--train_list', type=str, default=None,help='path to training pair list')
parser.add_argument('--valid_list', type=str, default=None,help='path to validating pair list')
parser.add_argument('--batch_size', type=int, default=512,
                    help='input batch size for training (default: 512)')
parser.add_argument('--is_gray', type=bool, default=False,
                    help='Transform input image to gray or not  (default: False)')
# Network
parser.add_argument('--weight_model', type=str, default='checkpoint/Mar02-00-34-21/CosFace_15_checkpoint.pth')
parser.add_argument('--weight_fc', type=str, default='checkpoint/Mar02-00-34-21/CosFace_15_checkpoint_classifier.pth')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--resume_fc', type=str, default='')
parser.add_argument('--s_weight', type=float, default=10.0)
# Classifier
parser.add_argument('--num_class', type=int, default=2622,
                    help='number of people(class)')
parser.add_argument('--classifier_type', type=str, default='MCP',
                    help='Which classifier for train. (MCP, AL, L)')
# LR policy
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--step_size', type=list, default=None,
                    help='lr decay step')  
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    metavar='W', help='weight decay (default: 0.0005)')
parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys(),
                    help='the number of settings and hyperparameters used in training')
# Common settings
parser.add_argument('--log_interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='',
                    help='path to save checkpoint')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='disables CUDA training')
parser.add_argument('--workers', type=int, default=8,
                    help='how many workers to load data')
parser.add_argument('--gpus', type=str, default='4,5,6,7')
parser.add_argument('--ngpus', type=int, default=4)
parser.add_argument('--d_name', type=str, default='')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


def main():
    # --------------------------------------model----------------------------------------
    model = net_sia.LResNet50E_IR_Sia(is_gray=args.is_gray)
    model_eval = net_sia.LResNet50E_IR_Sia(is_gray=args.is_gray)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    # 512 is dimension of feature
    classifier = {
        'MCP': layer.MarginCosineProduct(512, args.num_class),
        'AL' : layer.AngleLinear(512, args.num_class),
        'L'  : torch.nn.Linear(512, args.num_class, bias=False)
    }[args.classifier_type]

    classifier.load_state_dict(torch.load(args.weight_fc))

    print(os.environ['CUDA_VISIBLE_DEVICES'], args.cuda)
    
    pretrained = torch.load(args.weight_model) 
    pretrained_dict = pretrained['model_state_dict']
    model_dict = model.state_dict()
    model_eval_dict = model_eval.state_dict()
    for k, v in pretrained_dict.items():
        if k in model_dict: 
            model_dict[k].copy_(v)
        
    del pretrained
    del pretrained_dict
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        # classifier ckpt only save model info 
        classifier.load_state_dict(torch.load(args.resume_fc))
    print(model)
    model = torch.nn.DataParallel(model).to(device)
    model_eval = model_eval.to(device)
    classifier = classifier.to(device)
    

    args.run_name = utils.get_run_name()  
    output_dir = os.path.join(args.save_path, args.run_name.split("_")[0])
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
        valid_transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    train_loader = torch.utils.data.DataLoader(
        dset.ImageList(root=args.root_path, fileList=args.train_list,
                  transform=train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dset.ImageList(root=args.root_path, fileList=args.valid_list,
                  transform=valid_transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False)

    print('length of train Database: ' + str(len(train_loader.dataset)) + ' Batches: ' + str(len(train_loader)))
    print('length of valid Database: ' + str(len(val_loader.dataset)) + ' Batches: ' + str(len(val_loader)))
    print('Number of Identities: ' + str(args.num_class))
    # Get a batch of training data, (img, img_occ, label)
    ''' 
    inputs, inputs_occ, imgPair, targets = next(iter(train_loader)) 
    out = torchvision.utils.make_grid(inputs)
    out_occ = torchvision.utils.make_grid(inputs_occ)
    
    mean = torch.tensor((0.5,0.5,0.5), dtype=torch.float32)
    std = torch.tensor((0.5,0.5,0.5), dtype=torch.float32)
    utils.imshow(out, mean, std, title=str(targets))
    plt.savefig(output_dir + '/train.png')
    utils.imshow(out_occ, mean, std, title=str(targets))
    plt.savefig(output_dir + '/train_occ.png')
    '''
    #---------------------------------------params setting-----------------------------------  
    for name, param in model.named_parameters():
        if 'layer' in name or 'conv1' in name or 'bn1' in name or 'prelu1' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    print("Params to learn:")
    params_to_update = []
    params_to_stay = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            if 'sia' in name:
                params_to_update.append(param)
                print("Update \t", name)
            else:
                params_to_stay.append(param)
                print("Stay \t", name)

    for name, param in classifier.named_parameters():
        param.requires_grad = True
        params_to_stay.append(param)
        print("Stay \t", name)
    #--------------------------------loss function and optimizer-----------------------------
    cfg = configurations[args.config]
    criterion = torch.nn.CrossEntropyLoss().to(device)
    criterion2 = torch.nn.L1Loss(reduction='mean').to(device)
    optimizer = torch.optim.SGD([
                                    {'params': params_to_stay, 'lr': 0, 'weight_decay': 0, 'momentum': 0},
                                    {'params': params_to_update}
                                ],
                                lr=cfg['lr'],
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    start_epoch = 1
    if args.resume:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        start_epoch = checkpoint['epoch']
        del checkpoint
    # ----------------------------------------train----------------------------------------
    save_ckpt(model, 0, optimizer, output_dir + '/CosFace_0_checkpoint.pth') # Not resumed, pretrained~
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        train(train_loader, model, classifier, criterion, criterion2, optimizer, epoch, cfg['step_size'], cfg['lr'])
        save_ckpt(model, epoch, optimizer, output_dir + '/CosFace_' + str(epoch) + '_checkpoint.pth')
        print('Validating on valid set...')
        valid(val_loader, model_eval, output_dir + '/CosFace_' + str(epoch) + '_checkpoint.pth', classifier, criterion, criterion2)
    print('Finished Training')
    

def valid(val_loader, model, model_path, classifier, criterion, criterion2):
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    top1_clean = utils.AverageMeter()
    top1_occ = utils.AverageMeter()
    losses_clean = utils.AverageMeter()
    losses_occ = utils.AverageMeter()
    sia_losses = utils.AverageMeter()

    time_curr = time.time()
    time_curr0 = time.time()
    
    
    with torch.no_grad():
        for batch_idx, (data, data_occ, _, target) in enumerate(val_loader, 1):
            
            data, data_occ, target = data.to(device), data_occ.to(device), target.to(device)
            f_masked, focc_masked, output, output_occ, _, _ = model(data, data_occ)
            #f_masked_diff = torch.abs(f_masked - focc_masked)
            #temp = torch.sum(f_masked_diff) / (f_masked.size(0) * 512 * 7 * 6)
            
            sia_loss = criterion2(focc_masked, f_masked) # / (f_masked.size(0) * 512 * 7 * 6)
            sia_losses.update(sia_loss.item(),data.size(0))

            score = torch.mm(output, classifier.weight.t())
            prec1 = utils.accuracy(score.data, target.data)
            top1_clean.update(prec1[0].item(),data.size(0))

            score_occ = torch.mm(output_occ, classifier.weight.t())
            prec1_occ = utils.accuracy(score_occ.data, target.data)
            top1_occ.update(prec1_occ[0].item(),data.size(0))
            
            output = classifier(output, target)
            loss_clean = criterion(output, target)
            output_occ = classifier(output_occ, target)
            loss_occ = criterion(output_occ, target)

            loss = 0.5 * loss_clean + 0.5 * loss_occ + args.s_weight * sia_loss
            #print(loss_clean.item(), loss_occ.item(), sia_loss.item())
            if np.isnan(float(loss.item())):
                raise ValueError('loss is nan while validating')
            losses_clean.update(loss_clean.item(),data.size(0))
            losses_occ.update(loss_occ.item(),data.size(0))
            
            
            if batch_idx % 100 == 0:
                time_used = time.time() - time_curr
                print_with_time(
                        'Valid : [{}/{} ({:.0f}%)], SiaLoss: {:.6f}, CleanLoss: {:.6f}, OccLoss: {:.6f}, Prec: {:.4f}, Prec_occ: {:.4f}, Elapsed time: {:.4f}s({} iters)'.format(
                        batch_idx * len(data), len(val_loader.dataset), 100. * batch_idx / len(val_loader),
                        sia_losses.avg, losses_clean.avg, losses_occ.avg, top1_clean.avg, top1_occ.avg, time_used, 100)
                )
                time_curr = time.time()
        
        # end of validation print
        time_used = time.time() - time_curr0
        print_with_time(
                'Valid Summary : SiaLoss: {:.6f}, CleanLoss: {:.6f}, OccLoss: {:.6f}, Prec: {:.4f}, Prec_occ: {:.4f}, Elapsed time: {:.4f}s'.format(
                    sia_losses.avg, losses_clean.avg, losses_occ.avg, top1_clean.avg, top1_occ.avg, time_used)
        )



def train(train_loader, model, classifier, criterion, criterion2, optimizer, epoch, step_size, base_lr):

    for name, module in model.module.named_children(): 
        if name in ['sia']:
            module.train(True)
        else:
            module.train(False)
    print_with_time('Epoch {} start training'.format(epoch))
    time_curr = time.time()
    time_curr0 = time.time()
    loss_display = 0.0
    
    top1_clean = utils.AverageMeter()
    top1_occ = utils.AverageMeter()
    losses_clean = utils.AverageMeter()
    losses_occ = utils.AverageMeter()
    sia_losses = utils.AverageMeter()
    
    for batch_idx, (data, data_occ, _, target) in enumerate(train_loader, 1):
        
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        adjust_learning_rate(optimizer, iteration, step_size, base_lr)
        data, data_occ, target = data.to(device), data_occ.to(device), target.to(device)
        f_masked, focc_masked, output, output_occ, f_diff, out = model(data, data_occ)

        #----------------------------------------Calculate statistics--------------------------------------#
        sia_loss = criterion2(focc_masked, f_masked) #/ (f_masked.size(0) * 512 * 7 * 6)
        sia_losses.update(sia_loss.item(),data.size(0)) 
        #f_masked_diff = torch.abs(f_masked - focc_masked)
        #temp = torch.sum(f_masked_diff) / (f_masked.size(0) * 512 * 7 * 6)
        #print(temp.item(),sia_loss.item(), f_masked.shape)

        score = torch.mm(output, classifier.weight.t())
        prec1 = utils.accuracy(score.data, target.data)
        top1_clean.update(prec1[0].item(),data.size(0))

        score_occ = torch.mm(output_occ, classifier.weight.t())
        prec1_occ = utils.accuracy(score_occ.data, target.data)
        top1_occ.update(prec1_occ[0].item(),data.size(0))
        
        output = classifier(output, target)
        loss_clean = criterion(output, target)
        losses_clean.update(loss_clean.item(),data.size(0))

        output_occ = classifier(output_occ, target)
        loss_occ = criterion(output_occ, target)
        losses_occ.update(loss_occ.item(),data.size(0))
        #---------------------------------------------Backward----------------------------------------------#
        loss = 0.5 * loss_clean + 0.5 * loss_occ + args.s_weight * sia_loss
        #print(loss_clean.item(), loss_occ.item(), sia_loss.item())
        if np.isnan(float(loss.item())):
            raise ValueError('loss is nan while validating')
        loss_display += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % 1600 == 0:
            print(out[0,0,:,:])
        if batch_idx % 100 == 0:
            time_used = time.time() - time_curr
            loss_display /= 100
            
            print_with_time(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]{}, SiaLoss: {:.6f}, ClsLoss: [{:.6f}/{:.6f}/{:.6f}], Prec: {:.4f}, Prec_occ: {:.4f}, Elapsed time: {:.4f}s, LR: {:.5f}/{:.5f}({} iters) INFO: [{}/{}/{}]'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    iteration, sia_losses.avg, losses_clean.avg, losses_occ.avg, loss_display, top1_clean.avg, top1_occ.avg, 
                    time_used, optimizer.param_groups[0]['lr'],optimizer.param_groups[1]['lr'],100, 'net2', args.s_weight,args.d_name)
            )
            time_curr = time.time()
            loss_display = 0.0
    time_used = time.time() - time_curr0
    print_with_time(
            'Train Summary : SiaLoss: {:.6f}, ClsLoss: [{:.6f}/{:.6f}], Prec: {:.4f}, Prec_occ: {:.4f}, Elapsed time: {:.4f}s, INFO: [{}/{}]'.format(
                sia_losses.avg, losses_clean.avg, losses_occ.avg, top1_clean.avg, top1_occ.avg, time_used, 'net2', args.s_weight)
    )


def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)


def adjust_learning_rate(optimizer, iteration, step_size, base_lr):
    """Sets the learning rate to the initial LR decayed by 10 each step size"""
    if iteration in step_size:
        lr = base_lr * (0.1 ** (step_size.index(iteration) + 1))
        print_with_time('Adjust base learning rate to {}'.format(lr))
        optimizer.param_groups[0]['lr'] = 0 
        optimizer.param_groups[1]['lr'] = lr
    else:
        pass


def save_ckpt(model, epoch, optimizer, save_name):
    """Save checkpoint""" 
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    torch.save({
        'epoch': epoch,
        'optim_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict()}, save_name)


if __name__ == '__main__': 
    print(args)
    main()
