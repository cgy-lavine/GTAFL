import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, Temp):
        super(DistillKL, self).__init__()
        self.T = Temp

    def forward(self, y_s, y_t):
        outputs = torch.log_softmax(y_s/self.T, dim=1)#学生网络的输出
        labels = torch.softmax(y_t/self.T, dim=1)#教师网络的输出
        outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
        loss = -torch.mean(outputs, dim=0, keepdim=False)
        return loss
    
class LwF_Loss(nn.Module):
    def __init__(self, Temp, lamda, num_cls, device):
        super().__init__()
        self.Temp = Temp
        self.lamda = lamda
        self.num_cls = num_cls
        self.device = device
        self.criterion_kd = DistillKL(Temp)
        self.criterion_cls = nn.CrossEntropyLoss()

    def forward(self, labels, teacher_pred, logits):
        pos_cls = torch.unique(labels).tolist()
        neg_cls = list(set([*range(self.num_cls)]).difference(set(pos_cls)))
        transformed_labels = torch.tensor([pos_cls.index(i) for i in labels]).to(logits.device)
        loss_cls = self.criterion_cls(logits[:, pos_cls], transformed_labels)
        loss_kd = self.criterion_kd(logits[:, neg_cls], teacher_pred[:, neg_cls])
        # print("pos_cls.{}".format(pos_cls))
        # print("neg_cls:{}".format(neg_cls))
        # print("loss_cls:{}".format(loss_cls))
        # print("loss_kd:{}".format(loss_kd))
        preds = torch.softmax(logits[:, pos_cls], dim=1)
        logs_preds = torch.log_softmax(logits[:, pos_cls], dim=1)
        loss_ent = torch.sum(-preds*logs_preds)

        # preds = torch.softmax(logits[:, pos_cls], dim=1)
        # logs_preds = torch.log_softmax(logits[:, pos_cls], dim=1)
        # loss_ent = torch.sum(-preds*logs_preds)
        # print(loss_cls, loss_kd, loss_ent)

        # if torch.isnan(loss_cls):
        #     print(pos_cls)
        #     print(logits[:, pos_cls].shape, labels.shape, max(transformed_labels))
        #     print(loss_cls)

        # select according to classes
        loss = loss_cls + self.lamda*loss_kd
        return loss, loss_cls, loss_kd
    
def create_loss(Temp, lamda, num_classes, device):
    return LwF_Loss(
        Temp = Temp,    # e.g., 2
        lamda = lamda,  # e.g., 1 
        num_cls = num_classes, 
        device = device
    )
