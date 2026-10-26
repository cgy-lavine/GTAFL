from torchvision import datasets
from torchvision.transforms import ToTensor, transforms
from options import args_parser
from Dataset.long_tailed_cifar10 import train_long_tail
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset, TensorDataset, get_class_num
from Dataset.sample_dirichlet import clients_indices
from Dataset.Gradient_matching_loss import match_loss
import numpy as np
from torch import stack, max, eq, no_grad, tensor, unsqueeze, split
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from Model.Resnet8 import ResNet_cifar
from tqdm import tqdm
import copy
import torch
import random
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset.param_aug import DiffAugment
from Dataset.TestAgnostic_CIFAR import TestAgnosticImbalanceCIFAR10DataLoader
import os
import math
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset
from Simclr_finetune import Simclr
from ensemble import test_training,test_validation

class Global(object):
    def __init__(self,
                 num_classes: int,
                 device: str,
                 global_distribution,
                 args,
                 num_of_feature):
        self.device = device
        self.num_classes = num_classes
        self.fedavg_acc = []
        self.fedavg_many = []
        self.fedavg_medium = []
        self.fedavg_few = []
        self.ft_acc = []
        self.ft_many = []
        self.ft_medium = []
        self.ft_few = []
        self.num_of_feature = num_of_feature
        self.feature_syn_uniform = torch.randn(size=(args.num_classes * self.num_of_feature, 256), dtype=torch.float,
                                       requires_grad=True, device=args.device)
        self.label_syn_uniform = torch.tensor([np.ones(self.num_of_feature) * i for i in range(args.num_classes)], dtype=torch.long,
                                      requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        self.optimizer_feature = SGD([self.feature_syn_uniform, ], lr=args.lr_feature)  # optimizer_img for synthetic data
        self.criterion = CrossEntropyLoss().to(args.device)
        self.syn_model = ResNet_cifar(resnet_size=8, scaling=4,
                                      save_activations=False, group_norm_num_groups=None,
                                      freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(device)
        self.feature_net_forward = nn.Linear(256, 10).to(args.device)
        self.feature_net_uniform = nn.Linear(256, 10).to(args.device)
        self.feature_net_backward = nn.Linear(256, 10).to(args.device)
        self.global_distribution = global_distribution

        #正向长尾分类头
        base_values = [self.num_of_feature] * args.num_classes  # 这是前面的100
        additional_values = [math.ceil(x * args.lada) for x in self.global_distribution]  # 后面的5到1数据
        self.num_feature_syn_distribution_forward = [base + add for base, add in zip(base_values, additional_values)]
        self.feature_syn_forward = torch.randn(size=(sum(self.num_feature_syn_distribution_forward), 256), dtype=torch.float,
                                       requires_grad=True, device=args.device)
        self.label_syn_forward = self.create_label_syn(self.num_feature_syn_distribution_forward)
        self.optimizer_feature_forward = SGD([self.feature_syn_forward, ], lr=args.lr_feature)  # optimizer_img for synthetic data
         #反向长尾分类头
        self.num_feature_syn_distribution_backward = self.num_feature_syn_distribution_forward[::-1]
        self.feature_syn_backward = torch.randn(size=(sum(self.num_feature_syn_distribution_backward), 256), dtype=torch.float,
                                       requires_grad=True, device=args.device)
        self.label_syn_backward = self.create_label_syn(self.num_feature_syn_distribution_backward)
        self.optimizer_feature_backward = SGD([self.feature_syn_backward, ], lr=args.lr_feature)  # optimizer_img for synthetic data

        self.aggregation_weight = torch.nn.Parameter(torch.FloatTensor(3), requires_grad=True)
        self.aggregation_weight.data.fill_(1/3) 
        self.optimizer_ensemble = SGD([self.aggregation_weight] ,lr=0.1,weight_decay=5e-4,momentum=0.9,nesterov=True)

    def create_label_syn(self,distribution):
        labels = []
        
        for class_index in range(self.num_classes):
            num_features = distribution[class_index]
            labels.extend([class_index]*num_features)
        
        return torch.tensor(labels, dtype=torch.long, requires_grad=False, device=self.device).view(-1)

    def update_feature_syn(self, args, global_params, list_clients_gradient,expert):#global params是重训的分类器+server端的feature层,list_clients_gradients是每个类的梯度
        if expert == "forward":
            feature_net_params = self.feature_net_forward.state_dict()#分类头
        elif expert == "uniform":
            feature_net_params = self.feature_net_uniform.state_dict()
        else:
            feature_net_params = self.feature_net_backward.state_dict()


        for name_param in reversed(global_params):
            if name_param == 'classifier.bias':
                feature_net_params['bias'] = global_params[name_param]
            if name_param == 'classifier.weight':
                feature_net_params['weight'] = global_params[name_param]
                break

        if expert == "forward":
            self.feature_net_forward.load_state_dict(feature_net_params)
            self.feature_net_forward.train()
            net_global_parameters = list(self.feature_net_forward.parameters())
        elif expert == "uniform":
            self.feature_net_uniform.load_state_dict(feature_net_params)
            self.feature_net_uniform.train()
            net_global_parameters = list(self.feature_net_uniform.parameters())
        else:
            self.feature_net_backward.load_state_dict(feature_net_params)
            self.feature_net_backward.train()
            net_global_parameters = list(self.feature_net_backward.parameters())

        
        gw_real_all = {class_index: [] for class_index in range(self.num_classes)}
        for gradient_one in list_clients_gradient:
            for class_num, gradient in gradient_one.items():
                gw_real_all[class_num].append(gradient)
        gw_real_avg = {class_index: [] for class_index in range(args.num_classes)}
        # aggregate the real feature gradients
        for i in range(args.num_classes):
            gw_real_temp = []
            list_one_class_client_gradient = gw_real_all[i]

            if len(list_one_class_client_gradient) != 0:
                weight_temp = 1.0 / len(list_one_class_client_gradient)
                for name_param in range(2):
                    list_values_param = []
                    for one_gradient in list_one_class_client_gradient:
                        list_values_param.append(one_gradient[name_param] * weight_temp)
                    value_global_param = sum(list_values_param)
                    gw_real_temp.append(value_global_param)
                gw_real_avg[i] = gw_real_temp
        # update the federated features.
        for ep in range(args.match_epoch):
            loss_feature = torch.tensor(0.0).to(args.device)
            feature_index = 0
            for c in range(args.num_classes):
                if len(gw_real_avg[c]) != 0:
                    
                    if expert == "forward":
                        feature_syn = self.feature_syn_forward[feature_index:feature_index+self.num_feature_syn_distribution_forward[c]].reshape((self.num_feature_syn_distribution_forward[c],256))
                        lab_syn = torch.ones((self.num_feature_syn_distribution_forward[c],),device=args.device,dtype=torch.long) * c
                        output_syn = self.feature_net_forward(feature_syn)
                        feature_index = feature_index+self.num_feature_syn_distribution_forward[c]

                    elif expert == "uniform":
                        feature_syn = self.feature_syn_uniform[c * self.num_of_feature:(c + 1) * self.num_of_feature].reshape((self.num_of_feature, 256))
                        lab_syn = torch.ones((self.num_of_feature,), device=args.device, dtype=torch.long) * c
                        output_syn = self.feature_net_uniform(feature_syn)#100*256扔进256*10时变成100*10
                    else:
                        feature_syn = self.feature_syn_backward[feature_index:feature_index+self.num_feature_syn_distribution_backward[c]].reshape((self.num_feature_syn_distribution_backward[c],256))
                        lab_syn = torch.ones((self.num_feature_syn_distribution_backward[c],),device=args.device,dtype=torch.long) * c
                        output_syn = self.feature_net_backward(feature_syn)
                        feature_index = feature_index+self.num_feature_syn_distribution_backward[c]
        
                    loss_syn = self.criterion(output_syn, lab_syn)
                    # compute the federated feature gradients of class c
                    gw_syn = torch.autograd.grad(loss_syn, net_global_parameters, create_graph=True)
                    loss_feature += match_loss(gw_syn, gw_real_avg[c], args)

                
            if expert == "forward":
                self.optimizer_feature_forward.zero_grad()
                loss_feature.backward()
                self.optimizer_feature_forward.step()
            elif expert == "uniform":
                self.optimizer_feature.zero_grad()
                loss_feature.backward()
                self.optimizer_feature.step()
            else:
                self.optimizer_feature_backward.zero_grad()
                loss_feature.backward()
                self.optimizer_feature_backward.step()

    def feature_re_train(self, fedavg_params, batch_size_local_training,expert):#re-train the classifier using federated features
        if expert == "forward":
            feature_syn_train_ft = copy.deepcopy(self.feature_syn_forward.detach())#1000*256
            label_syn_train_ft = copy.deepcopy(self.label_syn_forward.detach())#1000,每个class都是100个feature
        elif expert == "uniform":
            feature_syn_train_ft = copy.deepcopy(self.feature_syn_uniform.detach())
            label_syn_train_ft = copy.deepcopy(self.label_syn_uniform.detach())
        else:
            feature_syn_train_ft = copy.deepcopy(self.feature_syn_backward.detach())
            label_syn_train_ft = copy.deepcopy(self.label_syn_backward.detach())

        dst_train_syn_ft = TensorDataset(feature_syn_train_ft, label_syn_train_ft)
        ft_model = nn.Linear(256, 10).to(args.device)
        optimizer_ft_net = SGD(ft_model.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
        ft_model.train()
        for epoch in range(args.crt_epoch):
            trainloader_ft = DataLoader(dataset=dst_train_syn_ft,
                                        batch_size=batch_size_local_training,
                                        shuffle=True)
            for data_batch in trainloader_ft:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = ft_model(images)
                loss_net = self.criterion(outputs, labels)
                optimizer_ft_net.zero_grad()
                loss_net.backward()
                optimizer_ft_net.step()
        ft_model.eval()
        feature_net_params = ft_model.state_dict()
        for name_param in reversed(fedavg_params):
            if name_param == 'classifier.bias':
                fedavg_params[name_param] = feature_net_params['bias']
            if name_param == 'classifier.weight':
                fedavg_params[name_param] = feature_net_params['weight']
                break
        return copy.deepcopy(ft_model.state_dict()), copy.deepcopy(fedavg_params)

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):#list_num_local_data表示每个客户端有多少训练数据
        # fedavg
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def global_eval(self, fedavg_params, data_test, batch_size_test,):
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()

        num_corrects = 0
        num_data = 0

        # 初始化各类计数
        many_corrects = 0
        many_data = 0
        medium_corrects = 0
        medium_data = 0
        few_corrects = 0
        few_data = 0

        with torch.no_grad():
            test_loader = data_test
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)

                num_corrects += torch.sum(predicts == labels).item()
                num_data += len(labels)
                for label, predict in zip(labels.cpu(), predicts.cpu()):
                    if label in range(0, 3):  # Many shot
                        many_corrects += (predict == label).item()
                        many_data += 1
                    elif label in range(3, 6):  # Medium shot
                        medium_corrects += (predict == label).item()
                        medium_data += 1
                    elif label in range(6, 10):  # Few shot
                        few_corrects += (predict == label).item()
                        few_data += 1

                #num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            #accuracy = num_corrects / len(data_test)
         # 计算总体准确率
        accuracy = num_corrects / num_data if num_data else 0

        # 计算各类准确率
        many_accuracy = many_corrects / many_data if many_data else 0
        medium_accuracy = medium_corrects / medium_data if medium_data else 0
        few_accuracy = few_corrects / few_data if few_data else 0

        shot=[round(many_accuracy,4),round(medium_accuracy,4),round(few_accuracy,4)]
        #return accuracy
        return round(accuracy, 4), shot  # 返回总体准确率


    def download_params(self):
        return self.syn_model.state_dict()
    
class ClientSampler(Dataset):
    def __init__(self, dataset, indices, global_distribution,args):
        self.dataset = dataset
        self.indices = indices
        self.args = args
        self.labels = [self.dataset[i][1] for i in indices]
        self.class_counts = self._compute_class_counts()
        self.global_distribution = global_distribution

    def _compute_class_counts(self):
        class_counts = np.zeros(10)
        unique, counts = np.unique(self.labels, return_counts=True)
        class_counts[unique] = counts
        return class_counts
    
    def cosine_similarity(self,v1,v2):
        dot_product = np.dot(v1,v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        similarity = dot_product/(norm_v1*norm_v2)
        return similarity

    def get_client_sampler(self):#获取全局分布的倒数
        global_distribution = np.array(self.global_distribution)
        unique_labels = list(np.unique(self.labels))
        transformed_labels = torch.tensor([unique_labels.index(i) for i in self.labels])#把self.labels中的内容对应的unique_labels的下标提取出来
        filter_global_distribution = global_distribution[unique_labels]
        weights = 1. / filter_global_distribution#全局倒数
        local_distribution = np.delete(self.class_counts,np.where(self.class_counts == 0))
        # weights = 1. / local_distribution
        samples_weight = torch.tensor([weights[label] for label in transformed_labels])
    
        cossimilarity_value = self.cosine_similarity(global_distribution,self.class_counts)
        # 创建 WeightedRandomSampler
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
            
        return cossimilarity_value, sampler


class Local(object):
    def __init__(self,
                 data_all,
                 data_client,
                 global_distribuiton,
                 class_list: int,
                 client_distribution_list: int):
        args = args_parser()
        self.data_all = data_all
        self.data_client = data_client
        self.global_distribution = global_distribuiton
        

        self.device = args.device
        self.class_compose = class_list
        self.client_distribution_list = client_distribution_list
        self.criterion = CrossEntropyLoss().to(args.device)

        self.local_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(
            args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)

    def compute_gradient(self, global_params, args):#用重置过后的分类头计算梯度
        # compute C^k
        list_class, per_class_compose = get_class_num(self.class_compose)  # class组成  list_class 是类别比如说[0,1,2,6],per_class_composehh是每个类别的数量[300,289,110,1]
        images_all = []
        labels_all = []
        indices_class = {class_index: [] for class_index in list_class}

        images_all = [unsqueeze(self.data_client[i][0], dim=0) for i in range(len(self.data_client))]
        labels_all = [self.data_client[i][1] for i in range(len(self.data_client))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)#lab是类别,i是索引
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        # get random n images from class c
        def get_images(c, n):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]#不够32张就返回全部图片
            return images_all[idx_shuffle]

        self.local_model.load_state_dict(global_params)

        self.local_model.eval()
        self.local_model.classifier.train()
        net_parameters = list(self.local_model.classifier.parameters())
        criterion = CrossEntropyLoss().to(args.device)
        # gradients of all classes
        truth_gradient_all = {index: [] for index in list_class}
        truth_gradient_avg = {index: [] for index in list_class}

        # choose to repeat 10 times
        for num_compute in range(10):
            for c, num in zip(list_class, per_class_compose):
                img_real = get_images(c, args.batch_real)
                # transform
                if args.dsa:
                    # seed = int(time.time() * 1000) % 100000
                    seed = args.seed
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                feature_real, output_real = self.local_model(img_real)#output_real是分类的概率
                loss_real = criterion(output_real, lab_real)
                # compute the real feature gradients of class c
                gw_real = torch.autograd.grad(loss_real, net_parameters)#与classifier.parameters做梯度的计算
                gw_real = list((_.detach().clone() for _ in gw_real))
                truth_gradient_all[c].append(gw_real)
        for i in list_class:
            gw_real_temp = []
            gradient_all = truth_gradient_all[i]
            weight = 1.0 / len(gradient_all)#len(gradient_all)都是10
            for name_param in range(len(gradient_all[0])):
                list_values_param = []
                for client_one in gradient_all:
                    list_values_param.append(client_one[name_param] * weight)
                value_global_param = sum(list_values_param)
                gw_real_temp.append(value_global_param)
            # the real feature gradients of all classes
            truth_gradient_avg[i] = gw_real_temp
        return truth_gradient_avg

    def local_train(self, args, global_params):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        client_sampler = ClientSampler(self.data_all, self.client_distribution_list, self.global_distribution,args)#修改
        cossimilarity_value, sampler = client_sampler.get_client_sampler()#修改

        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        for _ in range(args.num_epochs_local_training):
            if args.global_sample == True:
                if cossimilarity_value > 0.90:
                    data_loader = DataLoader(dataset=self.data_client,
                                        batch_size=args.batch_size_local_training,
                                        shuffle=False,
                                        sampler=sampler
                                        )
                else:
                    data_loader = DataLoader(dataset=self.data_client,
                                        batch_size=args.batch_size_local_training,
                                        shuffle=True,
                                        )
            else:
                data_loader = DataLoader(dataset=self.data_client,
                                    batch_size=args.batch_size_local_training,
                                    shuffle=True,
                                    )
                

            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                images = transform_train(images)
                _, outputs = self.local_model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.local_model.state_dict()

def create_mutil_test(args):
    distrb = {
            'forward50': (0.02, False),
            'forward25': (0.04, False), 
            'forward10':(0.1, False),
            'forward5': (0.2, False),
            'forward2': (0.5, False),
            'uniform': (1,False),
            'backward2': (0.5, True),
            'backward5': (0.2, True),
            'backward10': (0.1, True),
            'backward25': (0.04, True),
            'backward50': (0.02, True),   
        }
    test_distribution_set = ["forward50",  "forward25", "forward10", "forward5", "forward2", "uniform",  "backward2", "backward5", "backward10", "backward25", "backward50"]
    max_accuracies = {key: 0 for key in distrb.keys()}
    max_acc_3shot_global = {key: {"head": 0, "middle": 0, "tail": 0} for key in distrb.keys()}
    mutil_test_acc = {key:[ ] for key in distrb.keys()}
    ori_max_acc =  0
    ori_max_acc_3shot_global = {key: {"head": 0, "middle": 0, "tail": 0} for key in distrb.keys()}
    test_loaders = {}
    for test_type in test_distribution_set:
        test_IF = distrb[test_type][0]
        reverse = distrb[test_type][1]
        data_global_test_LT = TestAgnosticImbalanceCIFAR10DataLoader(
                args.seed,
                data_dir=args.path_cifar10,
                test_type=test_type,
                batch_size=128,
                shuffle=False,
                training=False,
                num_workers=1,
                test_imb_factor=test_IF,
                imb_type='exp',
                reverse=reverse,
            )
        test_loaders[test_type] = data_global_test_LT.test_set()
    
    return test_loaders, max_accuracies,max_acc_3shot_global,mutil_test_acc

def experiment_logs_name(args):
    # if args.simclr == True:
    #     name = "lambda_{}_{}_seed{}_IF{}_non_iid{}.txt".format(args.lada, args.dataset_name, args.seed, args.imb_factor, args.non_iid_alpha)
    # else:
    #     name = "no_classifier_retraining_{}_seed{}_IF{}_non_iid{}.txt".format(args.dataset_name, args.seed, args.imb_factor, args.non_iid_alpha)
    # return name
    if args.simclr == True:
        if args.global_sample == True:
            name = "GTAFL_global_sample_{}_seed{}_IF{}_non_iid{}.txt".format(args.dataset_name, args.seed, args.imb_factor, args.non_iid_alpha)
        else:
            name = "GTAFL_no_sample_{}_seed{}_IF{}_non_iid{}.txt".format(args.dataset_name, args.seed, args.imb_factor, args.non_iid_alpha)
    else:
        name = "CReff_{}_seed{}_IF{}_non_iid{}.txt".format(args.dataset_name, args.seed, args.imb_factor, args.non_iid_alpha)
    return name

def create_test_data(args):
    distrb = {
        'uniform': (1,False),
        'forward50': (0.02, False),
        'forward25': (0.04, False), 
        'forward10':(0.1, False),
        'forward5': (0.2, False),
        'forward2': (0.5, False),
        'backward50': (0.02, True),
        'backward25': (0.04, True),
        'backward10': (0.1, True),
        'backward5': (0.2, True),
        'backward2': (0.5, True),
    }
    test_distribution_set = ["forward50",  "forward25", "forward10", "forward5", "forward2", "uniform",  "backward2", "backward5", "backward10", "backward25", "backward50"]  
    test_IF= distrb[args.test_distribution][0]
    reverse = distrb[args.test_distribution][1]
    data_global_test_LT = TestAgnosticImbalanceCIFAR10DataLoader(
            args.seed,
            args.path_cifar10,
            test_type = args.test_distribution,
            batch_size=args.simclr_batch_size,
            shuffle=False,
            training=False,
            num_workers=1,
            test_imb_factor= test_IF,
            imb_type= args.imb_type,
            reverse= reverse,
        )
    
    data_test_all = data_global_test_LT.test_set()
    unlabel_data_test =data_global_test_LT.train_set()

    return data_test_all, unlabel_data_test

def GTAFL():
    args = args_parser()
    random_state = np.random.RandomState(args.seed)

    experiment_log_name = experiment_logs_name(args)

    with open(experiment_log_name, 'a') as log_file:
        log_file.write(
            'imb_factor:{ib}, non_iid:{non_iid},global_sample:{global_sample}, simclr:{simclr},simclr_lr:{simclr_lr},simclr_epo:{simclr_epo},seed:{seed}\n'
            'lr_net:{lr_net}, lr_feature:{lr_feature}, num_of_feature:{num_of_feature},lada:{lada}\n'
            'match_epoch:{match_epoch}, re_training_epoch:{crt_epoch},data_fraction:{data_fraction}, test_distribution:{test_distribution}\n'.format(
                ib=args.imb_factor,
                non_iid=args.non_iid_alpha,
                global_sample = args.global_sample,
                simclr = args.simclr,
                simclr_lr = args.simclr_lr,
                simclr_epo = args.simclr_epo,
                seed = args.seed,
                lr_net=args.lr_net,
                lr_feature=args.lr_feature,
                num_of_feature=args.num_of_feature,
                lada = args.lada,
                match_epoch=args.match_epoch,
                crt_epoch=args.crt_epoch,
                data_fraction = args.data_fraction,
                test_distribution = args.test_distribution))
    # Load data
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
    # Distribute data
    if args.mutil_test == True:
        test_loaders, max_accuracies,max_acc_3shot_global,mutil_test_acc = create_mutil_test(args)
    

    list_label2indices = classify_label(data_local_training, args.num_classes)
    # heterogeneous and long_tailed setting
    global_distribution, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
   
    

    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                           args.num_clients, args.non_iid_alpha, args.seed)

    original_dict_per_client = show_clients_data_distribution(
        data_local_training, list_client2indices, args.num_classes)
    global_model = Global(num_classes=args.num_classes,
                          device=args.device,
                          global_distribution=global_distribution,
                          args=args,
                          num_of_feature=args.num_of_feature)
    total_clients = list(range(args.num_clients))
    indices2data = Indices2Dataset(data_local_training)

    data_test_all, unlabel_data_test = create_test_data(args)
    re_train_acc_forward = []
    re_train_acc_uniform = []
    re_train_acc_backward = []
    forward_shot_max =[]
    uniform_shot_max =[]   
    backward_shot_max =[]

    # 重训分类器
    temp_model_forward = nn.Linear(256, 10).to(args.device)
    temp_model_uniform = nn.Linear(256, 10).to(args.device)
    temp_model_backward = nn.Linear(256, 10).to(args.device)

    syn_params_forward = temp_model_forward.state_dict()
    syn_params_uniform = temp_model_uniform.state_dict()
    syn_params_backward = temp_model_backward.state_dict()

    max_accuracy_forward = 0 #修改,最大准确率
    max_accuracy_uniform = 0
    max_accuracy_backward = 0
    max_params = [None,None,None]


    simclr_writers = SummaryWriter(log_dir ="runs/simclr_loss" )

    for r in tqdm(range(1, args.num_rounds+1), desc='server-training'):
        global_params = global_model.download_params()
        syn_feature_params_forward = copy.deepcopy(global_params)#复制整个global网络，并且重置分类层
        syn_feature_params_uniform = copy.deepcopy(global_params)#复制整个global网络，并且重置分类层
        syn_feature_params_backward = copy.deepcopy(global_params)#复制整个global网络，并且重置分类层

        for name_param in reversed(syn_feature_params_forward):#换分类头
            if name_param == 'classifier.bias':
                syn_feature_params_forward[name_param] = syn_params_forward['bias']
            if name_param == 'classifier.weight':
                syn_feature_params_forward[name_param] = syn_params_forward['weight']
                break

        for name_param in reversed(syn_feature_params_uniform):
            if name_param == 'classifier.bias':
                syn_feature_params_uniform[name_param] = syn_params_uniform['bias']
            if name_param == 'classifier.weight':
                syn_feature_params_uniform[name_param] = syn_params_uniform['weight']
                break

        for name_param in reversed(syn_feature_params_backward):
            if name_param == 'classifier.bias':
                syn_feature_params_backward[name_param] = syn_params_backward['bias']
            if name_param == 'classifier.weight':
                syn_feature_params_backward[name_param] = syn_params_backward['weight']
                break

        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_clients_gradient_forward = []
        list_clients_gradient_uniform = []
        list_clients_gradient_backward = []

        list_dicts_local_params = []
        list_nums_local_data = []

        total_loss = 0.0  # 用于累加所有客户端的损失
        avg_loss_label = 0.0
        avg_loss_simclr = 0.0

        # local training
        for client in online_clients:
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))
            local_model = Local(data_all = data_local_training,
                                data_client=data_client,
                                global_distribuiton=global_distribution,
                                class_list=original_dict_per_client[client],
                                client_distribution_list=list_client2indices[client])#修改，为了sampler
            # compute the real feature gradients in local data
            truth_gradient_forward = local_model.compute_gradient(copy.deepcopy(syn_feature_params_forward), args)#重置过后的分类头,计算梯度,返回的是每个类针对分类头的梯度
            truth_gradient_uniform = local_model.compute_gradient(copy.deepcopy(syn_feature_params_uniform), args)
            truth_gradient_backward = local_model.compute_gradient(copy.deepcopy(syn_feature_params_backward), args)

            list_clients_gradient_forward.append(copy.deepcopy(truth_gradient_forward))
            list_clients_gradient_uniform.append(copy.deepcopy(truth_gradient_uniform))
            list_clients_gradient_backward.append(copy.deepcopy(truth_gradient_backward))
            # local update
            local_params = local_model.local_train(args, copy.deepcopy(global_params))
            list_dicts_local_params.append(copy.deepcopy(local_params))
        # aggregating local models with FedAvg
        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        global_model.update_feature_syn(args, copy.deepcopy(syn_feature_params_forward), list_clients_gradient_forward,expert = "forward")#第一个参数是重新训练过分类头的网络,第二个参数是梯度
        global_model.update_feature_syn(args, copy.deepcopy(syn_feature_params_uniform), list_clients_gradient_uniform,expert = "uniform")
        global_model.update_feature_syn(args, copy.deepcopy(syn_feature_params_backward), list_clients_gradient_backward,expert = "backward")#反向长尾

        syn_params_forward, ft_params_forward = global_model.feature_re_train(copy.deepcopy(fedavg_params), args.batch_size_local_training,expert="forward")#重新训练了分类头ft_params是分类头+训练好的feature层
        syn_params_uniform, ft_params_uniform = global_model.feature_re_train(copy.deepcopy(fedavg_params), args.batch_size_local_training,expert="uniform")#重新训练了分类头ft_params是分类头+训练好的feature层
        syn_params_backward, ft_params_backward = global_model.feature_re_train(copy.deepcopy(fedavg_params), args.batch_size_local_training,expert="backward")#重新训练了分类头ft_params是分类头+训练好的feature层

        if args.simclr == True:
            fedavg_params= Simclr(fedavg_params,unlabel_data_test,args,simclr_writers,r)
        
        one_re_train_acc_forward, forward_shot  = global_model.global_eval(ft_params_forward,  data_test_all, args.batch_size_test)
        one_re_train_acc_uniform, uniform_shot = global_model.global_eval(ft_params_uniform, data_test_all,args.batch_size_test)
        one_re_train_acc_backward, backward_shot= global_model.global_eval(ft_params_backward, data_test_all,args.batch_size_test)
        
        if args.mutil_test == True:
            for test_type, test_loader in test_loaders.items():
                # 评估当前测试集的准确率
                current_acc,acc_shot = global_model.global_eval(ft_params_uniform, test_loader, args.batch_size_test)
                mutil_test_acc[test_type].append(current_acc)
                
                # 检查并更新最高准确率
                if current_acc > max_accuracies[test_type]:
                    max_accuracies[test_type] = current_acc
                    # 更新 3-shot global 精确度
                    max_acc_3shot_global[test_type]["head"] = acc_shot[0] # 根据需要计算并更新
                    max_acc_3shot_global[test_type]["middle"] = acc_shot[1] # 根据需要计算并更新
                    max_acc_3shot_global[test_type]["tail"] = acc_shot[2] # 根据需要计算并更新

            # 输出当前测试集的结果
            #print(f"Test Type: {test_type}, Current Accuracy: {current_acc:.2f}, Max Accuracy: {max_accuracies[test_type]:.2f}")
        
        re_train_acc_forward.append(one_re_train_acc_forward)
        re_train_acc_uniform.append(one_re_train_acc_uniform)
        re_train_acc_backward.append(one_re_train_acc_backward)

        global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))

        if one_re_train_acc_forward > max_accuracy_forward:
            max_accuracy_forward = one_re_train_acc_forward
            max_params[0] = ft_params_forward
            forward_shot_max = forward_shot 
            

        if one_re_train_acc_uniform > max_accuracy_uniform:
            max_accuracy_uniform = one_re_train_acc_uniform
            max_params[1] = ft_params_uniform
            uniform_shot_max = uniform_shot

        if one_re_train_acc_backward > max_accuracy_backward:
            max_accuracy_backward = one_re_train_acc_backward
            max_params[2] = ft_params_backward
            backward_shot_max = backward_shot
            
    record_acc_shot(experiment_log_name,args.test_distribution,re_train_acc_forward,re_train_acc_uniform,re_train_acc_backward)
    if args.mutil_test == True:
        mutil_test_log_name = "Mutil_data_fraction_{}_cifar10_seed{}_trainIF{}non_iid{}_resnet{}.txt".format(args.data_fraction, args.seed, args.imb_factor,args.non_iid_alpha,args.resnet_size)
        with open(mutil_test_log_name, "a") as file:
            file.write(
            'imb_factor:{ib}, non_iid:{non_iid},global_sample:{global_sample}, simclr:{simclr},simclr_lr:{simclr_lr},simclr_epo:{simclr_epo},seed:{seed}\n'
            'lr_net:{lr_net}, lr_feature:{lr_feature}, num_of_feature:{num_of_feature},lada:{lada}\n'
            'match_epoch:{match_epoch}, re_training_epoch:{crt_epoch},resnet_size:{resnet_size},data_fraction:{data_fraction}, test_distribution:{test_distribution}\n'.format(
                ib=args.imb_factor,
                non_iid=args.non_iid_alpha,
                global_sample = args.global_sample,
                simclr = args.simclr,
                simclr_lr = args.simclr_lr,
                simclr_epo = args.simclr_epo,
                seed = args.seed,
                lr_net=args.lr_net,
                lr_feature=args.lr_feature,
                num_of_feature=args.num_of_feature,
                lada = args.lada,
                match_epoch=args.match_epoch,
                crt_epoch=args.crt_epoch,
                resnet_size = args.resnet_size,
                data_fraction = args.data_fraction,
                test_distribution = args.test_distribution))
            file.write("\n")  # 添加换行

            # 打印 max_accuracies
            file.write("\nMax Accuracies:\n")
            for test_type, max_acc in max_accuracies.items():
                file.write(f"{test_type}: {max_acc:.4f}\n")

            # 打印 max_acc_3shot_global
            file.write("\nMax 3-shot Global Accuracies:\n")
            for test_type, shots in max_acc_3shot_global.items():
                file.write(f"{test_type}: Head: {shots['head']}, Middle: {shots['middle']}, Tail: {shots['tail']}\n")
            
            # 打印 mutil_test_acc
            file.write("Mutil Test Accuracies:\n")
            for test_type, acc_list in mutil_test_acc.items():
                file.write(f"{test_type}: {acc_list}\n")

    print("forward_max_acc:{},many:{} medium:{} few:{}".format(max_accuracy_forward,forward_shot_max[0],forward_shot_max[1],forward_shot_max[2]))
    print("uniform_max_acc:{},many:{} medium:{} few:{}".format(max_accuracy_uniform,uniform_shot_max[0],uniform_shot_max[1],uniform_shot_max[2]))
    print("backward_max_acc:{},many:{} medium:{} few:{}".format(max_accuracy_backward,backward_shot_max[0],backward_shot_max[1],backward_shot_max[2]))

    # 保存最终训练的模型权重
    if args.save_model_weights == True:
        save_model_weights(args, max_params, max_accuracy_forward, max_accuracy_uniform, max_accuracy_backward)

    with open(experiment_log_name, 'a') as log_file:
        log_file.write("forward_max_acc:{},many:{} medium:{} few:{}\n".format(max_accuracy_forward,forward_shot_max[0],forward_shot_max[1],forward_shot_max[2]))
        log_file.write("uniform_max_acc:{},many:{} medium:{} few:{}\n".format(max_accuracy_uniform,uniform_shot_max[0],uniform_shot_max[1],uniform_shot_max[2]))
        log_file.write("backward_max_acc:{},many:{} medium:{} few:{}\n".format(max_accuracy_backward,backward_shot_max[0],backward_shot_max[1],backward_shot_max[2]))
        log_file.write("\n")  # 添加空行以分隔不同实验的记录

    aggregation_weight = torch.nn.Parameter(torch.FloatTensor(3), requires_grad=True)
    aggregation_weight.data.fill_(1/3)
    optimizer_ensemble = SGD([aggregation_weight] ,lr=0.1,weight_decay=5e-4,momentum=0.9,nesterov=True)
    global_model.aggregation_weight = test_training(unlabel_data_test,
                                                    max_params[0],max_params[1],max_params[2],
                                                    aggregation_weight,
                                                    optimizer_ensemble, args)
    
    ensemble_accuracy, ensemble_shot = test_validation(data_test_all,
                                        max_params[0],max_params[1],max_params[2],
                                        global_model.aggregation_weight,args)
    print("-------------------------ensemble_accuracy:{} many:{} medium:{} few:{} aggregation_weight{}".format(ensemble_accuracy, ensemble_shot[0], ensemble_shot[1], ensemble_shot[2], global_model.aggregation_weight))
    with open(experiment_log_name, 'a') as log_file:
            log_file.write(
                "-------------------------ensemble_name:{} accuracy:{} many:{} medium:{} few:{} aggregation_weight{}\n".format(
                    args.test_distribution,
                    ensemble_accuracy,
                    ensemble_shot[0],
                    ensemble_shot[1],
                    ensemble_shot[2],
                    global_model.aggregation_weight,
                )
            )
            log_file.write("----------------------------------------------------------------------------------------------------------------\n")


def record_acc_shot(expepriment_log_name,test_distribution,re_train_acc_forward,re_train_acc_uniform,re_train_acc_backward):
    with open(expepriment_log_name,'a') as log_file:
        log_file.write("------------test_Imbalance_Factor-------------{}\n".format(test_distribution))
        log_file.write("------------forward-------------{}\n".format(re_train_acc_forward))
        log_file.write("------------uniform-------------{}\n".format(re_train_acc_uniform))
        log_file.write("------------backward-------------{}\n".format(re_train_acc_backward))

def save_model_weights(args, max_params, max_accuracy_forward, max_accuracy_uniform, max_accuracy_backward):
    """保存最终训练的模型权重到本地文件"""
    import os

    # 创建二级目录结构
    save_dir = "saved_models"
    global_sample_dir = f"global_sample_{args.global_sample}"
    test_distribution_dir = f"test_distribution_{args.test_distribution}"

    # 创建完整路径
    full_save_dir = os.path.join(save_dir, global_sample_dir, test_distribution_dir)
    if not os.path.exists(full_save_dir):
        os.makedirs(full_save_dir)

    # 生成文件名（简化，因为目录已经包含了global_sample和test_distribution信息）
    base_name = f"model_seed{args.seed}"

    # 保存三个专家的模型权重
    model_names = ["forward", "uniform", "backward"]
    accuracies = [max_accuracy_forward, max_accuracy_uniform, max_accuracy_backward]

    for i, (model_name, params, acc) in enumerate(zip(model_names, max_params, accuracies)):
        if params is not None:
            filename = f"{base_name}_{model_name}_acc{acc:.4f}.pth"
            filepath = os.path.join(full_save_dir, filename)
            torch.save(params, filepath)

    # 保存实验配置信息
    config = {
        'seed': args.seed,
        'global_sample': args.global_sample,
        'test_distribution': args.test_distribution,
        'imb_factor': args.imb_factor,
        'non_iid_alpha': args.non_iid_alpha,
        'num_classes': args.num_classes,
        'max_accuracies': {
            'forward': max_accuracy_forward,
            'uniform': max_accuracy_uniform,
            'backward': max_accuracy_backward
        }
    }

    config_filename = f"{base_name}_config.pkl"
    config_filepath = os.path.join(full_save_dir, config_filename)
    import pickle
    with open(config_filepath, 'wb') as f:
        pickle.dump(config, f)

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = False
    
    GTAFL()


