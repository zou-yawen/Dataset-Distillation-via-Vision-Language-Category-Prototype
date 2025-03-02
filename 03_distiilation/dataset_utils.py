import os
import sys
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, index
def load_dataset(args):
    # Obtain dataloader
    transform_train = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
    ])
    if args.dataset == 'cifar10':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.491, 0.482, 0.447), (0.202, 0.199, 0.201))
        ])
        trainset = IndexedImageFolder(root=args.data_dir + "/train", 
                                        transform=transform_train)

    elif args.dataset == 'cifar100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.486, 0.441), (0.267, 0.256, 0.276))
        ])
        trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=False,
                                    transform=transform_train)

    elif args.dataset == 'imagenet':
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = IndexedImageFolder(root=args.data_dir + "/train", 
                                        transform=transform_train)
                       
    elif args.dataset == 'tiny_imagenet':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        trainset = IndexedImageFolder(root=args.data_dir + "/train", 
                                        transform=transform_train)



    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False
    )

    path_all = [path for path, _ in trainset.samples]

    return trainloader, path_all


def gen_label_list(args):
    # obtain label-prompt list
    with open(args.label_file_path, "r") as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        line = line.strip()
        label = line.split('\t')[0]
        labels.append(label)
    
    return labels
