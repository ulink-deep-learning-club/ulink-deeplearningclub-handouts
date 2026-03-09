import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    对比损失实现

    Args:
        margin: 间隔参数，默认1.0
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Args:
            output1, output2: 样本对特征 [B, D]
            label: 相似性标签 [B], 1表示同类, 0表示不同类

        Returns:
            loss: 标量损失值
        """
        # 计算欧氏距离
        euclidean_distance = F.pairwise_distance(
            output1, output2, keepdim=True
        )  # [B, 1]

        # 正样本对损失: label * d^2
        loss_positive = label * torch.pow(euclidean_distance, 2)

        # 负样本对损失: (1-label) * max(0, m-d)^2
        loss_negative = (1 - label) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0),
            2
        )

        # 平均损失
        loss_contrastive = torch.mean(loss_positive + loss_negative)

        return loss_contrastive
