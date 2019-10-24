from __future__ import print_function
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import socket
from datetime import datetime

def get_id_label_map(meta_file):
    keys = []
    values = []
    with open(meta_file, 'r') as f:
        for i, class_name in enumerate(f):
            class_name = class_name.strip()
            keys.append(class_name)
            values.append(i)

    id_label_dict = dict(zip(keys, values))
    return id_label_dict

def get_id_labels_map(meta_file_ori, meta_file_new):
    id_label_dict_ori = get_id_label_map(meta_file_ori)
    keys = []
    label_maps = []
    with open(meta_file_new, 'r') as f:
        for i, class_name in enumerate(f):
            class_name = class_name.strip()
            keys.append(class_name)
            # name new_label old_label
            label_maps.append([i, id_label_dict_ori[class_name]])
    id_label_dict_new = dict(zip(keys, label_maps))

    return id_label_dict_new


def generate_img_file_list(img_folder, list_save_path, dsource):
    f = open(list_save_path, 'a+')
    for root, dirs, files in os.walk(img_folder):
        class_name = root.split("/")[-1]
        class_name = class_name.split("\\")[-1]
        for k, file_name in enumerate(files):
            img_file_path = dsource + '/' + class_name + '/' + file_name
            f.write(img_file_path)
            f.write("\n")

    f.close()


def Untransform_VGG(img, mean_bgr):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img += mean_bgr
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]
    return img
def Untransform(img, meann, stdd): 
    img.to(torch.float32) 
    img = img.mul_(stdd[:,None,None]).add_(meann[:,None,None])
    img = img.numpy() 
    img = img.transpose(1, 2, 0)
    return img
    

def imshow(inp, meann, stdd, title=None):
    """Imshow for Tensor."""
    inp = Untransform(inp, meann, stdd)
  
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    # plt.show()


def load_state_dict(model, fname):
    pretrained_weight = torch.load(fname)
    own_state = model.state_dict()
    pre_para_list = []
    own_para_list = []
    for p_id, p_name in enumerate(pretrained_weight):
        pre_para_list.append(p_name)

    for p_id, p_name in enumerate(own_state):
        if p_id != 26 and p_id != 27:
            own_para_list.append(p_name)

    param_map = dict(zip(pre_para_list, own_para_list))
    # write into file to check
    f = open('param_map2.txt', 'w')
    f.write('Pretrained \t:\tSiamese \n')
    for key in param_map:
        key_value = key + '\t:\t' + param_map[key]
        f.write(key_value)
        f.write("\n")
    f.close()
    for pre_name in pretrained_weight:
        own_name = param_map[pre_name]
        if own_name in own_state:
            try:
                own_state[own_name].copy_(pretrained_weight[pre_name])
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, whose dimensions in the model are {} and whose ' \
                    'dimensions in the checkpoint are {}.'.format(own_name, own_state[own_name].size(), \
                                                                  pretrained_weight[pre_name].size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(own_name))

def load_state_dict_bn(model, fname):
    pretrained_weight = torch.load(fname)
    own_state = model.state_dict()

    pre_para_list = []
    own_para_list = []
    for p_id, p_name in enumerate(pretrained_weight):
        pre_para_list.append(p_name)

    for p_id, p_name in enumerate(own_state):
        # conv_s w conv_s b bn_w bn_b bn_runningmean bn_running_var bn_nums_batch
        if p_id != 26 and p_id != 27 and p_id != 28 and p_id != 29 and p_id != 30 and p_id != 31 and p_id != 32:
            own_para_list.append(p_name)

    param_map = dict(zip(pre_para_list, own_para_list))
    # write into file to check
    f = open('param_map2.txt', 'w')
    f.write('Pretrained \t:\tSiamese \n')
    for key in param_map:
        key_value = key + '\t:\t' + param_map[key]
        f.write(key_value)
        f.write("\n")
    f.close()
    for pre_name in pretrained_weight:
        own_name = param_map[pre_name]
        if own_name in own_state:
            try:
                own_state[own_name].copy_(pretrained_weight[pre_name])
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, whose dimensions in the model are {} and whose ' \
                    'dimensions in the checkpoint are {}.'.format(own_name, own_state[own_name].size(), \
                                                                  pretrained_weight[pre_name].size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(own_name))


def init_vgg_en(model, weight_file, id_label_dict_new):
    pretrained_weight = torch.load(weight_file)
    own_state = model.state_dict()

    pre_para_list = []
    own_para_list = []
    for p_id, p_name in enumerate(pretrained_weight):
        pre_para_list.append(p_name)

    for p_id, p_name in enumerate(own_state):
        own_para_list.append(p_name)

    param_map = dict(zip(own_para_list, pre_para_list))
    ori_id = []
    new_id = []
    for keys, values in id_label_dict_new.items():
        new_id.append(int(values[0]))
        ori_id.append(int(values[1]))

    for own_name in own_state:
        pre_name = param_map[own_name]
        if (pre_name in pretrained_weight) and ('fc8' not in pre_name):
            try:
                own_state[own_name].copy_(pretrained_weight[pre_name])
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, whose dimensions in the model are {} and whose ' \
                    'dimensions in the checkpoint are {}.'.format(own_name, own_state[own_name].size(), \
                                                                  pretrained_weight[pre_name].size()))
        elif (pre_name in pretrained_weight) and ('fc8.weight' in pre_name):
            pre_weight = pretrained_weight[pre_name]  # torch.Size([2622, 4096])
            own_weight = own_state[own_name]  # torch.Size([2000, 4096])
            own_weight[new_id, :] = pre_weight[ori_id, :]

        elif (pre_name in pretrained_weight) and ('fc8.bias' in pre_name):
            pre_bias = pretrained_weight[pre_name]
            own_bias = own_state[own_name]
            own_bias[new_id] = pre_bias[ori_id]

        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(own_name))


def init_sia_mask(model, weight_file, id_label_dict_new):
    pretrained_weight = torch.load(weight_file)
    own_state = model.state_dict()

    pre_para_list = []
    own_para_list = []
    for p_id, p_name in enumerate(pretrained_weight):
        pre_para_list.append(p_name)

    for p_id, p_name in enumerate(own_state):
        own_para_list.append(p_name)

    param_map = dict(zip(own_para_list, pre_para_list))
    ori_id = []
    new_id = []
    for keys, values in id_label_dict_new.items():
        new_id.append(int(values[0]))
        ori_id.append(int(values[1]))

    for own_name in own_state:
        pre_name = param_map[own_name]
        print(own_name, pre_name)
        if (pre_name in pretrained_weight) and ('fc8' not in pre_name):
            try:
                own_state[own_name].copy_(pretrained_weight[pre_name])
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, whose dimensions in the model are {} and whose ' \
                    'dimensions in the checkpoint are {}.'.format(own_name, own_state[own_name].size(), \
                                                                  pretrained_weight[pre_name].size()))
        elif (pre_name in pretrained_weight) and ('fc8.weight' in pre_name):
            pre_weight = pretrained_weight[pre_name]  # torch.Size([2622, 4096])
            own_weight = own_state[own_name]  # torch.Size([2000, 4096])
            own_weight[new_id, :] = pre_weight[ori_id, :]

        elif (pre_name in pretrained_weight) and ('fc8.bias' in pre_name):
            pre_bias = pretrained_weight[pre_name]
            own_bias = own_state[own_name]
            own_bias[new_id] = pre_bias[ori_id]

        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(own_name))

def init_sia_mask_from_vggen(model, weight_file):
    checkpoint = torch.load(weight_file)
    pretrained_weight = checkpoint['model_state_dict']
    own_state = model.state_dict()

    pre_para_list = []
    own_para_list = []
    for p_id, p_name in enumerate(pretrained_weight):
        pre_para_list.append(p_name)
    
    #print(pre_para_list)

    for p_id, p_name in enumerate(own_state):
        own_para_list.append(p_name)

    param_map = dict(zip(own_para_list, pre_para_list))
    print('Check the correspondency:sia vgg_en\n')
    for own_name in own_state:
        pre_name = param_map[own_name]
        print(own_name, pre_name)
        if pre_name in pretrained_weight:
            try:
                own_state[own_name].copy_(pretrained_weight[pre_name])
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, whose dimensions in the model are {} and whose ' \
                    'dimensions in the checkpoint are {}.'.format(own_name, own_state[own_name].size(), \
                                                                  pretrained_weight[pre_name].size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(own_name))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0) 
    output_sorted, pred = output.topk(maxk, 1, True, True)
    pred = pred.t() 
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True) 
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_run_name():
    """ A unique name for each run """
    return datetime.now().strftime(
        '%b%d-%H-%M-%S') + '_' + socket.gethostname()

