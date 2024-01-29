import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate
import config as c
from multi_transform_loader import ImageFolderMultiTransform


def get_random_transforms(args):
    augmentative_transforms = []
    if args.transf_rotations:
        augmentative_transforms += [transforms.RandomRotation(180)]
    if args.transf_brightness > 0.0 or args.transf_contrast > 0.0 or args.transf_saturation > 0.0:
        augmentative_transforms += [transforms.ColorJitter(brightness=args.transf_brightness,
                                                           contrast=args.transf_contrast,
                                                           saturation=args.transf_saturation)]
    tfs = [transforms.Resize(args.img_size)] \
          + augmentative_transforms\
          + [transforms.ToTensor(), transforms.Normalize(args.norm_mean, args.norm_std)]
    transform_train = transforms.Compose(tfs)
    return transform_train

def get_fixed_transforms(args):
    cust_rot = lambda x: rotate(x, args.degrees, False, False, None)
    augmentative_transforms = [cust_rot]
    if args.transf_brightness > 0.0 or args.transf_contrast > 0.0 or args.transf_saturation > 0.0:
        augmentative_transforms += [transforms.ColorJitter(brightness=args.transf_brightness,
                                                           contrast=args.transf_contrast,
                                                           saturation=args.transf_saturation)]
    tfs = [transforms.Resize(args.img_size)]\
          + augmentative_transforms\
          + [transforms.ToTensor(),transforms.Normalize(args.norm_mean,args.norm_std)]
    return transforms.Compose(tfs)


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def get_loss(z, jac):
    '''check equation 4 of the paper why this makes sense - oh and just ignore the scaling here'''
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]


def load_datasets(dataset_path, class_name, args):

    def target_transform(target):
        return class_perm[target]

    data_dir_train = os.path.join(dataset_path, class_name, 'train')
    data_dir_test = os.path.join(dataset_path, class_name, 'test')

    classes = os.listdir(data_dir_test)
    if 'good' not in classes:
        print('There should exist a subdirectory "good". Read the doc of this function for further information.')
        exit()
    classes.sort()
    class_perm = list()
    class_idx = 1
    for cl in classes:
        if cl == 'good':
            class_perm.append(0)
        else:
            class_perm.append(class_idx)
            class_idx += 1 # from 1 to n_classes-1

    transform_train = get_random_transforms(args)
    trainset = ImageFolderMultiTransform(data_dir_train,
                                         transform=transform_train,
                                         n_transforms=args.n_transforms,
                                         transf_brightness=args.transf_brightness,
                                         transf_contrast=args.transf_contrast,
                                         transf_saturation=args.transf_saturation,)
    testset = ImageFolderMultiTransform(data_dir_test,
                                        transform=transform_train,
                                        target_transform=target_transform, # class index
                                        n_transforms=args.n_transforms_test,
                                        transf_brightness=args.transf_brightness,
                                        transf_contrast=args.transf_contrast,
                                        transf_saturation=args.transf_saturation,)
    return trainset, testset


def make_dataloaders(trainset, testset, batch_size, batch_size_test) :
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True,
                                             batch_size=batch_size_test,
                                             shuffle=True,
                                             drop_last=False)
    return trainloader, testloader


def preprocess_batch(data):
    '''move data to device and reshape image'''
    inputs, labels = data
    inputs, labels = inputs.to(c.device), labels.to(c.device)
    inputs = inputs.view(-1, *inputs.shape[-3:])
    return inputs, labels
