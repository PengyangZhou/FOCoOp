import torch
import numpy as np


# https://github.com/FrancescoPinto/RegMixup/blob/main/models/regmixup.py
def mixup_data(x, y, soft_y, alpha=1.0, mix_strategy="mixup"):
    """Returns mixed inputs, pairs of targets, and lambda."""

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if lam > 0.5:
        lam = 1 - lam  # [0-0.5]
    if (
        mix_strategy == "mixup"
        or mix_strategy == "cutmix"
        or mix_strategy == "manimix"
        or mix_strategy == "geomix"
        or mix_strategy == "manimixrev"
    ):
        batch_size = x.size()[0]
        index = torch.arange(batch_size - 1, -1, -1).to(y.device)
        mask = y == y[index]
        while mask.any():
            swap_with = torch.randperm(batch_size).to(y.device)

            index[mask] = index[swap_with[mask]]

            mask = y == y[index]
        soft_ya, soft_yb = soft_y, soft_y[index]
        if mix_strategy == "mixup":
            mixed_x = lam * x + (1 - lam) * x[index]
        elif mix_strategy == "cutmix":
            bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
            mixed_x = x.clone()
            mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        elif mix_strategy == "manimix":
            mixed_x = lam * x + (1 - lam) * x[index]
            mixed_x = mixed_x / mixed_x.norm(dim=-1, keepdim=True)
        elif mix_strategy == "manimixrev":
            mixed_x = lam * x + (1 - lam) * x[index]
            mixed_x = x[index] - (mixed_x - x[index])
            mixed_x = mixed_x / mixed_x.norm(dim=-1, keepdim=True)
            ood_label = soft_y.new_zeros((soft_y.size(0), soft_y.size(1)))
            ood_label[:, -1] = 1
            soft_ya = ood_label
        elif mix_strategy == "geomix":
            mixed_x = sph_inter(x, x[index], lam)
    elif mix_strategy == "manimix_wccm" or mix_strategy == "wccm":
        image_features, text_feats = x
        batch_size = image_features.size()[0]
        text_num = text_feats.size()[1]
        random_indices = torch.randint(0, text_num, (batch_size,)).to(y.device)
        selected_text_feat = text_feats[y, random_indices, :]
        mixed_x = lam * image_features + (1 - lam) * selected_text_feat
        mixed_x = mixed_x / mixed_x.norm(dim=-1, keepdim=True)
        soft_ya, soft_yb = soft_y, soft_y
    elif mix_strategy == "mani_cccm":
        image_features, text_feats = x
        batch_size = image_features.size()[0]
        class_num, text_num = text_feats.size()[0], text_feats.size()[1]
        # data mix cross class.
        random_class_indices = torch.randint(0, class_num - 1, (batch_size,)).to(y.device)
        random_class_indices = torch.where(random_class_indices >= y, random_class_indices + 1, random_class_indices)
        random_indices = torch.randint(0, text_num, (batch_size,)).to(y.device)
        selected_text_feat = text_feats[random_class_indices, random_indices, :]  # same class.
        mixed_x = lam * image_features + (1 - lam) * selected_text_feat  # within class, cross modal.
        mixed_x = mixed_x / mixed_x.norm(dim=-1, keepdim=True)
        soft_text_y = torch.zeros_like(soft_y)
        soft_text_y[torch.arange(batch_size).to(y.device), random_class_indices] = 1
        soft_ya, soft_yb = soft_y, soft_text_y
    elif mix_strategy == "geomix_wccm":
        image_features, text_feats = x  # 128*512; 1000*7*512; 128
        batch_size = image_features.size()[0]
        text_num = text_feats.size()[1]
        random_indices = torch.randint(0, text_num, (batch_size,)).to(y.device)
        selected_text_feat = text_feats[y, random_indices, :]  # same class.
        mixed_x = sph_inter(image_features, selected_text_feat, lam)
        soft_ya, soft_yb = soft_y, soft_y
    elif mix_strategy == "geo_cc_cm":
        image_features, text_feats = x  # 128*512; 1000*7*512; 128
        batch_size = image_features.size()[0]
        class_num, text_num = text_feats.size()[0], text_feats.size()[1]
        # data mix cross class.
        random_class_indices = torch.randint(0, class_num - 1, (batch_size,)).to(y.device)
        random_class_indices = torch.where(random_class_indices >= y, random_class_indices + 1, random_class_indices)
        random_indices = torch.randint(0, text_num, (batch_size,)).to(y.device)
        selected_text_feat = text_feats[random_class_indices, random_indices, :]  # same class.
        mixed_x = sph_inter(image_features, selected_text_feat, lam)

        soft_text_y = torch.zeros_like(soft_y)
        soft_text_y[torch.arange(batch_size).to(y.device), random_class_indices] = 1
        soft_ya, soft_yb = soft_y, soft_text_y
    else:
        raise NotImplementedError

    ################# adding jusdgement, whether mixed data belong to the same label.
    return mixed_x, soft_ya, soft_yb, lam


# https://github.com/changdaeoh/multimodal-mixup
## can not self-mix, since torch.sin(theta)=0 will lead nan.
def sph_inter(a, b, s):
    theta = torch.acos((a * b).sum(dim=[1])).view(a.shape[0], 1)
    n1 = torch.sin(s * theta) / torch.sin(theta) * a
    n2 = torch.sin((1 - s) * theta) / torch.sin(theta) * b
    return n1 + n2


def regmixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
