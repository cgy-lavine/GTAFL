import argparse
import os
import torch
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image
from PIL import ImageFilter 
import torchvision
from .Imbalance_cifar import IMBALANCECIFAR10

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class  TestAgnosticImbalanceCIFAR10DataLoader(DataLoader):
    """
    Imbalance Cifar10 Data Loader
    """
    def __init__(self,seed ,data_dir,test_type, batch_size, shuffle=True, num_workers=1, training=True, balanced=False, retain_epoch_size=True, imb_type='exp', imb_factor=0.01, test_imb_factor=0, reverse=False,):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            normalize,
        ])
        test_trsfm = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_trsfm) # test set
        
        # if training:
        #     dataset = IMBALANCECIFAR10(data_dir, train=True, download=True, transform=train_trsfm, imb_type=imb_type, imb_factor=imb_factor)
        #     val_dataset = test_dataset
        # else: 
        print("------------------test_imbalance_F:{}".format(test_imb_factor))
        dataset = IMBALANCECIFAR10(data_dir, train=True, download=True, transform=train_trsfm, imb_type=imb_type, imb_factor=test_imb_factor, reverse=reverse) 
        train_dataset = IMBALANCECIFAR10(data_dir, train=False, download=True, transform= TwoCropsTransform(train_trsfm), imb_type=imb_type, imb_factor=test_imb_factor, reverse=reverse) 
        val_dataset = IMBALANCECIFAR10(data_dir, train=False, download=True, transform=test_trsfm, imb_type=imb_type, imb_factor=test_imb_factor, reverse=reverse) 
        
        # import dill
                # 保存索引到文件
        # if test_type == 'forward':
        #     file_path = os.path.join("transfer_dataset", "cifar10-LT_test_forward_seed{}".format(seed))
        #     with open(file_path, 'wb') as f:
        #         dill.dump(val_dataset, f)
    

        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 10
 
        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list

        
        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size, 
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs) # Note that sampler does not apply to validation set

 
    def train_set(self):
        return DataLoader(dataset=self.train_dataset, shuffle=True, **self.init_kwargs)

    def test_set(self):
        return DataLoader(dataset=self.val_dataset, shuffle=False, **self.init_kwargs) 
    
    def test_creff(self):
        return self.val_dataset

