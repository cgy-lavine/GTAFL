import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from torch.optim import SGD
from torch import stack, max, eq, no_grad, tensor, unsqueeze, split
from torch.optim import SGD
from Model.Resnet8 import ResNet_cifar
from tqdm import tqdm
from NTXentLoss import NTXentLoss
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from PIL import ImageFilter 
import random

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
def unlabel_data_transform(): #the sample as the Testagnostic_cifar10
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
    
    return test_trsfm


def Simclr(fedavg_params,unlabel_data_test_LT,args,simclr_writers,r):#平衡,正向,反向
    # model_all = []
    # model_para = [max_ft_params, max_ft_params2, max_ft_params3]
    device = args.device 
    epochs = args.simclr_epo#CIFAR10 = 10
    
    temperature = 0.2
    #for i in tqdm(range(0, 3)):
    model = ResNet_cifar(resnet_size=args.resnet_size, scaling=4,
            save_activations=False, group_norm_num_groups=None,
            freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
    
    # model_all.append(model)
    # model_all[i].load_state_dict(model_para[i])
    model.load_state_dict(fedavg_params)
    classifier = model.classifier
    model_simclr = GlobalSimclr(model).to(device)
    optimizer = optim.SGD(model_simclr.backbone.parameters(), lr=args.simclr_lr)
    ntxent_loss = NTXentLoss(device=device, temperature=temperature, use_cosine_similarity=True)

    total_loss_r = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for j, (data, _) in enumerate(unlabel_data_test_LT):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device) 

            projections1 = model_simclr(data[0])
            projections2 = model_simclr(data[1])

            #batch_size = projections1.size(0)
            # labels = torch.arange(batch_size).to(projections1.device)

            # 计算相似度矩阵
            # similarity_matrix = torch.mm(projections1, projections2.t()) / temperature
            loss = ntxent_loss(projections1,projections2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        total_loss_r += epoch_loss
    
    simclr_writers.add_scalar(f'simclr/R_Total_Loss', total_loss_r, r)
    model.classifier = classifier
    model_para = copy.deepcopy(model.state_dict())

    return model_para

class GlobalSimclr(nn.Module):
    def __init__(self, ResNet_cifar, projection_dim = 128):
        super(GlobalSimclr,self).__init__()
        self.backbone = ResNet_cifar
        feature_dim  = int(64 * ResNet_cifar.scaling * ResNet_cifar.layer3[0].expansion)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        features, _ = self.backbone(x)  # 忽略分类器的输出
        projections = self.projection_head(features)
        return projections
        #定义模型结构

class SimCLRResNet(nn.Module):
    def __init__(self, ResNet_cifar, projection_dim=128):
        super(SimCLRResNet, self).__init__()
        self.backbone = ResNet_cifar  # 使用传入的 ResNet 作为 backbone
        feature_dim  = int(64 * ResNet_cifar.scaling * ResNet_cifar.layer3[0].expansion)
        
        # 分类器 (用于有监督学习)
        self.classifier = ResNet_cifar.classifier 
        # SimCLR 投影头 (用于无监督学习)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        # Backbone 提取特征
        features, _ = self.backbone(x)
        
        # 分类器输出（用于有监督学习）
        logits = self.classifier(features)
        
        # SimCLR 投影头输出（用于对比学习）
        projections = self.projector(features)
        
        return features, logits, projections