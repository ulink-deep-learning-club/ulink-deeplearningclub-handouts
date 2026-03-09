import torch

def online_hard_negative_mining(anchor, positive, negative):
    """
    在线难例挖掘

    Args:
        anchor: 锚点特征 [B, D]
        positive: 正样本特征 [B, D]
        negative: 负样本特征 [B, D]

    Returns:
        hardest_positive_idx: 最难正样本索引
        hardest_negative_idx: 最难负样本索引
    """
    # Hardest positive: 同类别中距离最远
    d_ap = torch.norm(anchor - positive, dim=1)
    hardest_positive_idx = torch.argmax(d_ap)

    # Hardest negative: 不同类别中距离最近
    d_an = torch.norm(anchor - negative, dim=1)
    hardest_negative_idx = torch.argmin(d_an)

    return hardest_positive_idx, hardest_negative_idx
