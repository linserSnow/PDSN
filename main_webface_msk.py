from __future__ import print_function
from __future__ import division
import argparse
import os
import time

import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

cudnn.benchmark = True


import _init_paths
import net_msk
import datasets.dataset_pair_msk as dset
import lfw_eval_msk
import lfw_occ_eval_msk
import layer
import utils


# Training settings
parser = argparse.ArgumentParser(description='PyTorch CosFace')

# DATA
parser.add_argument('--root_path', type=str, default='/home/lingxuesong/data/CASIA-WebFace/',
                    help='path to root path of images')
parser.add_argument('--lfw_path', type=str, default='/home/lingxuesong/data/lfw/lfw-112X96_occ_msk_c2_mean_mix/',
                    help='path to root path of images')
parser.add_argument('--database', type=str, default='WebFace')
parser.add_argument('--train_list', type=str, default=None,
                    help='path to training list')
parser.add_argument('--batch_size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--is_gray', type=bool, default=False,
                    help='Transform input image to gray or not  (default: False)')
# Network
parser.add_argument('--weight_model', type=str, default='checkpoint/vggface1/Mar02-00-34-21/CosFace_15_checkpoint.pth')
parser.add_argument('--weight_fc', type=str, default='checkpoint/webface/Mar01-09-40-04/CosFace_29_checkpoint_classifier.pth')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--resume_fc', type=str, default='')
parser.add_argument('--clean_mask_flag', type=int, default=0)
# Classifier
parser.add_argument('--num_class', type=int, default=None,
                    help='number of people(class)')
parser.add_argument('--classifier_type', type=str, default='MCP')
# LR policy
parser.add_argument('--epochs', type=int, default=15,
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.1)')
parser.add_argument('--lr_freeze', type=float, default=0.1)
parser.add_argument('--step_size', type=list, default=None,
                    help='lr decay step')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    metavar='W', help='weight decay (default: 0.0005)')
# Common settings
parser.add_argument('--log_interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='checkpoint/',
                    help='path to save checkpoint')
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='disables CUDA training')
parser.add_argument('--workers', type=int, default=4,
                    help='how many workers to load data')
parser.add_argument('--gpus', type=str, default='0,1,2,3')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.database is 'WebFace':
    args.num_class = 10572
    args.step_size = [8000, 18000] # 452722  [5 10] 0.001 begin
else:
    raise ValueError("NOT SUPPORT DATABASE! ")


def main():
    # --------------------------------------model----------------------------------------
    model = net_msk.LResNet50E_IR(is_gray=args.is_gray)
    model_eval = net_msk.LResNet50E_IR(is_gray=args.is_gray)

    args.run_name = utils.get_run_name()
    output_dir = os.path.join(args.save_path, args.run_name.split("_")[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 512 is dimension of feature
    classifier = {
        'MCP': layer.MarginCosineProduct(512, args.num_class),
        'AL' : layer.AngleLinear(512, args.num_class),
        'L'  : torch.nn.Linear(512, args.num_class, bias=False)
    }[args.classifier_type]

    # load pretrained weight
    pretrained = torch.load(args.weight_model)
    model.load_state_dict(pretrained['model_state_dict'])
    model_eval.load_state_dict(pretrained['model_state_dict'])
    classifier.load_state_dict(torch.load(args.weight_fc))
    del pretrained
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(torch.load(args.resume_fc))
        print("Resume from epoch: {}".format(start_epoch))

    model = torch.nn.DataParallel(model).to(device)
    model_eval = model_eval.to(device)
    classifier = classifier.to(device)
    print(model)
    #model.module.save(output_dir + '/CosFace_0_checkpoint.pth')

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
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    print('length of train Database: ' + str(len(train_loader.dataset)) + '  Batches: ' + str(len(train_loader)))
    print('Number of Identities: ' + str(args.num_class))

    # --------------------------------params setting-----------------------------
    print("Params to learn:")
    params_to_update = []
    params_to_freeze = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            if 'fc' in name: # or 'layer4' in name:
                params_to_update.append(param)
                print("Update \t", name)
            else:
                params_to_freeze.append(param)
                print("Freeze \t", name)
    for name, param in classifier.named_parameters():
        param.requires_grad = True
        params_to_update.append(param)
        print("Update \t", name)

    # --------------------------------loss function and optimizer-----------------------------
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD([
                                    {'params': params_to_freeze, 'lr':args.lr * args.lr_freeze}, 
                                    {'params': params_to_update}
                                ],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
    # ----------------------------------------train----------------------------------------
    save_ckpt(model, 0, optimizer, output_dir + '/CosFace_0_checkpoint.pth')
    
    lfw_occ_eval_msk.eval(model_eval, args.lfw_path, output_dir + '/CosFace_0_checkpoint.pth', args.is_gray)
    lfw_eval_msk.eval(model_eval,output_dir + '/CosFace_0_checkpoint.pth', args.is_gray)
    for epoch in range(start_epoch, args.epochs + 1):
        train(train_loader, model, classifier, criterion, optimizer, epoch)
        save_ckpt(model, epoch, optimizer, output_dir + '/CosFace_' + str(epoch) + '_checkpoint.pth')
        torch.save(classifier.state_dict(), output_dir + '/CosFace_' + str(epoch) + '_checkpoint_classifier.pth') 
        lfw_eval_msk.eval(model_eval, output_dir + '/CosFace_' + str(epoch) + '_checkpoint.pth', args.is_gray)
        lfw_occ_eval_msk.eval(model_eval, args.lfw_path, output_dir + '/CosFace_' + str(epoch) + '_checkpoint.pth', args.is_gray)
    print('Finished Training')
    

def train(train_loader, model, classifier, criterion, optimizer, epoch):
    model.train()
    print_with_time('Epoch {} start training'.format(epoch))
    time_curr = time.time()
    loss_display = 0.0
    losses_clean = utils.AverageMeter()
    losses_occ = utils.AverageMeter()

    mask_ones = torch.ones(512, 7, 6)

    for batch_idx, (data, data_occ, _, masks, target) in enumerate(train_loader, 1):
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        adjust_learning_rate(optimizer, iteration, args.step_size)

        mask1 = mask_ones.expand(data.size(0), -1, -1, -1)

        data, data_occ, target = data.to(device), data_occ.to(device), target.to(device)
        masks = masks.to(device)
        # compute output
        if args.clean_mask_flag == 1: 
            output = model(data, masks)
        else:
            mask1 = mask1.to(device)
            output = model(data, mask1)
        output = classifier(output, target)
        loss_clean = criterion(output, target)

        output_occ = model(data_occ, masks)
        output_occ = classifier(output_occ, target)
        loss_occ = criterion(output_occ, target)
        
        losses_clean.update(loss_clean.item(),data.size(0))
        losses_occ.update(loss_occ.item(),data.size(0))
        
        loss = 0.5 * loss_clean + 0.5 * loss_occ
        loss_display += loss.item()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            if args.classifier_type is 'MCP':
                INFO = ' Freeze: {:.4f}, C_M_Flag: {} [INFO:WebMsk]'.format(args.lr_freeze, args.clean_mask_flag)
            elif args.classifier_type is 'AL':
                INFO = ' lambda: {:.4f}'.format(classifier.lamb)
            else:
                INFO = ''
            print_with_time(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.6f}/{:.6f}/{:.6f}, Elapsed time: {:.4f}s, LR: {:.6f}/{:.6f}({} iters)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    iteration, losses_clean.avg, losses_occ.avg, loss_display, time_used,
                    optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],args.log_interval) + INFO
            )
            time_curr = time.time()
            loss_display = 0.0
        torch.cuda.empty_cache()


def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)


def adjust_learning_rate(optimizer, iteration, step_size):
    """Sets the learning rate to the initial LR decayed by 10 each step size"""
    if iteration in step_size:
        lr = args.lr * (0.1 ** (step_size.index(iteration) + 1))
        print_with_time('Adjust learning rate to {}'.format(lr))
        optimizer.param_groups[0]['lr'] = args.lr_freeze * lr 
        optimizer.param_groups[1]['lr'] = lr
    else:
        pass

def save_ckpt(model, epoch, optimizer, save_name):
    """Save checkpoint""" 
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    
    torch.save({
        'epoch': epoch,
        # 'arch': self.model.__class__.__name__,
        'optim_state_dict': optimizer.state_dict(),
        'model_state_dict': model.state_dict()}, save_name)

if __name__ == '__main__':
    print(args)
    main()
