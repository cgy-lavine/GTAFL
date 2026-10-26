import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(
            use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity
    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 *
                    batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 *
                    batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)


    @staticmethod
    
    
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        batch_size = zis.size(0)
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(
            representations, representations)
        
        mask = self._get_correlated_mask(batch_size).to(self.device)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        negatives = similarity_matrix[mask].view(
            2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)

# import torch

# class NTXentLoss(torch.nn.Module):
#     def __init__(self, device, temperature=0.5, use_cosine_similarity=True):
#         super(NTXentLoss, self).__init__()
#         self.temperature = temperature
#         self.device = device
#         self.use_cosine_similarity = use_cosine_similarity
#         self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

#     def forward(self, zis, zjs):
#         batch_size = zis.size(0)

#         # 将两个视图的表示合并
#         representations = torch.cat([zis, zjs], dim=0).to(self.device)

#         # 计算相似度矩阵
#         if self.use_cosine_similarity:
#             similarity_matrix = torch.nn.functional.cosine_similarity(
#                 representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
#         else:
#             similarity_matrix = torch.mm(representations, representations.t())

#         # 创建掩码
#         mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
#         positive_indices = torch.arange(batch_size) + batch_size

#         # 正样本相似度
#         l_pos = similarity_matrix[torch.arange(batch_size), positive_indices]
#         r_pos = similarity_matrix[positive_indices, torch.arange(batch_size)]
#         positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

#         # 负样本相似度
#         negatives = similarity_matrix[~mask].view(2 * batch_size, -1)

#         logits = torch.cat([positives, negatives], dim=1)
#         logits /= self.temperature

#         labels = torch.zeros(2 * batch_size, dtype=torch.long, device=self.device)
#         loss = self.criterion(logits, labels)
#         return loss / (2 * batch_size)