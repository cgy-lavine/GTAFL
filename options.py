import argparse
import os
from Dataset.param_aug import ParamDiffAug


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--num_epochs_local_training', type=int, default=10)  #
    parser.add_argument('--batch_size_local_training', type=int, default=32)
    parser.add_argument('--match_epoch', type=int, default=100)
    parser.add_argument('--crt_epoch', type=int, default=300)
    parser.add_argument('--batch_real', type=int, default=32)
    parser.add_argument('--num_of_feature', type=int, default=100)
    parser.add_argument('--lr_feature', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_size_test', type=int, default=500)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.02, type=float, help='imbalance factor')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--save_path', type=str, default=os.path.join(path_dir, 'result/'))
    parser.add_argument('--method', type=str, default='DSA', help='DC/DSA')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    # FedProx
    parser.add_argument('--mu', type=float, default=0.01)
    
    # FedAvgM
    parser.add_argument('--init_belta', type=float, default=0.97)
    parser.add_argument("--test_distribution", type=str,default='backward50')
    parser.add_argument('--lada',type= float, default= 0.01)#ci
    parser.add_argument("--global_sample", action='store_true', default=False)#
    parser.add_argument("--simclr", action='store_true', default=False)#
    parser.add_argument("--simclr_batch_size", type=int, default=128)#
    parser.add_argument("--simclr_epo", type = int, default= 10)
    parser.add_argument("--simclr_lr", type = float, default= 1e-3)#cifar10 = 1e-3
    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--resnet_size", type= int, default=8)
    parser.add_argument("--mutil_test",action='store_true',default=False)#when it is ture, simclr = false
    parser.add_argument("--vote",action='store_true',default=True)
    parser.add_argument('--test_training_epos', type=int, default=5)
    parser.add_argument('--data_fraction', type=float, default=1)
    parser.add_argument('--save_model_weights', action='store_true', default=False)
    parser.add_argument('--no_classifier_retraining', action='store_true', default=False)
    args = parser.parse_args()
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    return args
