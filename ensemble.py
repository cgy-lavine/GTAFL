import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch
import numpy as np
from Model.Resnet8 import ResNet_cifar
import matplotlib.pyplot as plt

def test_training(train_data_loader,fed_param,fed_param1,fed_param2,  aggregation_weight, optimizer, args):
    model  = ResNet_cifar(resnet_size=args.resnet_size, scaling=4,
                save_activations=False, group_norm_num_groups=None,
                freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
    model1 = ResNet_cifar(resnet_size=args.resnet_size, scaling=4,
                save_activations=False, group_norm_num_groups=None,
                freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
    model2 = ResNet_cifar(resnet_size=args.resnet_size, scaling=4,
                save_activations=False, group_norm_num_groups=None,
                freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
    
    model.load_state_dict(fed_param)
    model1.load_state_dict(fed_param1)
    model2.load_state_dict(fed_param2)

    model.eval()
    model1.eval()
    model2.eval()
    aggregation_weight_list = []
    aggregation_weight_list.append(aggregation_weight) 
 
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    for _ in range(args.test_training_epos):
        for i, (data, _) in enumerate(train_data_loader):
            data[0] = data[0].to(args.device)
            data[1] = data[1].to(args.device) 
            _,expert_logits_output0 = model(data[0])
            _,expert1_logits_output0 = model1(data[0])
            _,expert2_logits_output0 = model2(data[0])
            _,expert_logits_output1 = model(data[1])
            _,expert1_logits_output1 = model1(data[1])
            _,expert2_logits_output1 = model2(data[1])
            aggregation_softmax = torch.nn.functional.softmax(aggregation_weight) # softmax for normalization
            aggregation_output0 = aggregation_softmax[0].to(args.device) * expert_logits_output0 + aggregation_softmax[1].to(args.device) * expert1_logits_output0 + aggregation_softmax[2].to(args.device) * expert2_logits_output0
            aggregation_output1 = aggregation_softmax[0].to(args.device) * expert_logits_output1 + aggregation_softmax[1].to(args.device) * expert1_logits_output1 + aggregation_softmax[2].to(args.device) * expert2_logits_output1
            softmax_aggregation_output0 = F.softmax(aggregation_output0, dim=1) 
            softmax_aggregation_output1 = F.softmax(aggregation_output1, dim=1)
            
            # SSL loss: similarity maxmization
            cos_similarity = cos(softmax_aggregation_output0, softmax_aggregation_output1).mean()
            ssl_loss =  cos_similarity
            loss =  - ssl_loss 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            aggregation_weight_list.append(aggregation_softmax)

        if aggregation_softmax[0] < 0.05 or aggregation_softmax[1]<0.05 or aggregation_softmax[2]<0.05:
            break
            #losses.update(ssl_loss, data[0].shape[0])
        
    aggregation_softmax = torch.nn.functional.softmax(aggregation_weight, dim=0).detach().cpu().numpy()
    #print("----------------------aggregation:{}".format(aggregation_weight_list))
    aggregation_softmax_tensor = torch.tensor([np.round(aggregation_softmax[0], decimals=4),
                                               np.round(aggregation_softmax[1], decimals=4),
                                               np.round(aggregation_softmax[2], decimals=4)],requires_grad=False)

    #return  np.round(aggregation_softmax[0], decimals=2), np.round(aggregation_softmax[1], decimals=2), np.round(aggregation_softmax[2], decimals=2)
    return aggregation_softmax_tensor


def _head_mid_tail_ranges(args):
    return (0, 3), (3, 6), (6, 10)


def test_validation(data_loader,fed_param,fed_param1,fed_param2, aggregation_weight, args):
    model  = ResNet_cifar(resnet_size=args.resnet_size, scaling=4,
                save_activations=False, group_norm_num_groups=None,
                freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
    model1 = ResNet_cifar(resnet_size=args.resnet_size, scaling=4,
                save_activations=False, group_norm_num_groups=None,
                freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
    model2 = ResNet_cifar(resnet_size=args.resnet_size, scaling=4,
                save_activations=False, group_norm_num_groups=None,
                freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
    model.load_state_dict(fed_param)
    model1.load_state_dict(fed_param1)
    model2.load_state_dict(fed_param2)
    
    model.eval()
    model1.eval()
    model2.eval()
    
    # # 导入可视化函数
    # if args.picture == True:
    #     from visualize_pseudo_labels import compare_pseudo_labels
        
    #     # 准备模型列表和名称
    #     models = [model, model1, model2]
    #     model_names = ["Forward Expert", "Uniform Expert", "Backward Expert"]
        
    #     # 调用比较函数进行可视化
    #     all_counts = compare_pseudo_labels(
    #         models=models,
    #         test_loader=data_loader,
    #         num_classes=args.num_classes,
    #         model_names=model_names,
    #         device=args.device
    #     )
        
    #     # 保存图表
    #     plt.savefig(f'{args.test_distribution}_pseudo_labels.png')
    #     plt.close()
    
    #aggregation_weight.requires_grad = False
    r_many, r_med, r_few = _head_mid_tail_ranges(args)

    many_corrects = many_data = 0
    medium_corrects = medium_data = 0
    few_corrects = few_data = 0

    if args.vote == True:
        with torch.no_grad():
            num_corrects = 0
            num_data = 0
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(args.device), target.to(args.device)
                
                # 获取三个模型的预测
                _, expert_logits_output = model(data)
                _, expert1_logits_output = model1(data)
                _, expert2_logits_output = model2(data)
                # 原来的基于权重的集成方式（已注释）
                # aggregation_softmax = torch.nn.functional.softmax(aggregation_weight) # softmax for normalization
                # aggregation_output = aggregation_softmax[0] * expert_logits_output + aggregation_softmax[1] * expert1_logits_output + aggregation_softmax[2] * expert2_logits_output
                # _, predicts = torch.max(aggregation_output, 1)
                # num_corrects += sum(torch.eq(predicts.cpu(), target.cpu())).item()
                # 获取每个模型的预测类别
                _, pred1 = torch.max(expert_logits_output, 1)
                _, pred2 = torch.max(expert1_logits_output, 1)
                _, pred3 = torch.max(expert2_logits_output, 1)
                
                # 将预测结果堆叠在一起
                predictions = torch.stack([pred1, pred2, pred3])
                
                # 对每个样本进行投票
                final_predictions = torch.mode(predictions, dim=0)[0]
                
                # 计算正确预测的数量
                num_corrects += sum(torch.eq(final_predictions.cpu(), target.cpu())).item()
                num_data += len(target)

                for label, predict in zip(target.cpu(), final_predictions.cpu()):
                    lab = int(label.item())
                    ok = (predict == label).item()
                    if r_many[0] <= lab < r_many[1]:
                        many_corrects += ok
                        many_data += 1
                    elif r_med[0] <= lab < r_med[1]:
                        medium_corrects += ok
                        medium_data += 1
                    elif r_few[0] <= lab < r_few[1]:
                        few_corrects += ok
                        few_data += 1
    else:
        with torch.no_grad():
            num_corrects = 0
            num_data = 0
            for i, (data, target) in enumerate(tqdm(data_loader)):
                data, target = data.to(args.device), target.to(args.device)
                _,expert_logits_output = model(data)
                _,expert1_logits_output = model1(data)
                _,expert2_logits_output = model2(data)
                aggregation_softmax = torch.nn.functional.softmax(aggregation_weight) # softmax for normalization
                aggregation_output = aggregation_softmax[0] * expert_logits_output+ aggregation_softmax[1] *expert1_logits_output + aggregation_softmax[2] * expert2_logits_output
                _, predicts = torch.max(aggregation_output, 1)
                num_corrects += sum(torch.eq(predicts.cpu(), target.cpu())).item()
                num_data += len(target)

                for label, predict in zip(target.cpu(), predicts.cpu()):
                    lab = int(label.item())
                    ok = (predict == label).item()
                    if r_many[0] <= lab < r_many[1]:
                        many_corrects += ok
                        many_data += 1
                    elif r_med[0] <= lab < r_med[1]:
                        medium_corrects += ok
                        medium_data += 1
                    elif r_few[0] <= lab < r_few[1]:
                        few_corrects += ok
                        few_data += 1

    accuracy = num_corrects / num_data if num_data else 0.0
    many_accuracy = many_corrects / many_data if many_data else 0.0
    medium_accuracy = medium_corrects / medium_data if medium_data else 0.0
    few_accuracy = few_corrects / few_data if few_data else 0.0
    shot = [round(many_accuracy, 4), round(medium_accuracy, 4), round(few_accuracy, 4)]

    return round(accuracy, 4), shot  # 与 Global.global_eval 一致：总体 + [head, middle, tail]
    
