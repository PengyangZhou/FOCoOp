import torch
import torch.nn.functional as F
from torch import nn


class BiDRO(nn.Module):
    def __init__(self, logit_scale, num_iteration, num_ex_prompt, gamma_1=10, gamma_2=10):
        super(BiDRO, self).__init__()
        self.logit_scale = logit_scale.exp()
        self.num_ex_prompt = num_ex_prompt
        self.num_iteration = num_iteration
        # self.gamma=10
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.gamma_init = 10
        self.rho = 10
        self.beta = 0.005
        self.strength = 1
        self.epsilon = 1e-8

    def generate_adl(self, step, img_features, id_features, ood_features, targets, optimizer):
        num_classes = id_features.shape[0]
        id_noise = torch.rand_like(id_features) * 1e-4

        id_noise.requires_grad_(True)
        prev_loss = None
        id_features = id_features.detach()
        ood_features = ood_features.detach()

        for i in range(step):
            id_noise.requires_grad_(True)
            id_features_hat = id_features + id_noise
            logits_hat = self.logit_scale * img_features @ torch.cat([id_features_hat, ood_features], dim=0).T
            cs_loss = F.cross_entropy(logits_hat, targets)
            regularization = self.gamma_1 * torch.mean(torch.norm(id_noise, p=2, dim=1))
            adv_loss = cs_loss - regularization
            grads = torch.autograd.grad(adv_loss, [id_noise])[0]
            # grads /= (grads**2).sum(-1).sqrt().unsqueeze(1) + 1e-16
            norm_grads = torch.norm(grads, p=2, dim=-1, keepdim=True) + 1e-8
            grads /= norm_grads
            grads[torch.isinf(grads) | torch.isnan(grads)] = 0
            grad_clip_value = 1.0
            grads = torch.clamp(grads, -grad_clip_value, grad_clip_value)
            id_noise = id_noise.detach() + self.strength * grads.detach()
            optimizer.zero_grad()
            if prev_loss is not None and abs(adv_loss.item() - prev_loss) < self.epsilon:
                break
            prev_loss = adv_loss.item()

        ood_noise = torch.rand_like(ood_features) * 1e-4
        ood_noise.requires_grad_(True)
        id_features_hat = id_features + id_noise.detach()
        prev_loss = None
        for i in range(step):
            ood_noise.requires_grad_(True)
            ood_features_hat = ood_features + ood_noise
            logits_hat = self.logit_scale * img_features @ torch.cat([id_features_hat, ood_features_hat], dim=0).T
            logits_id = logits_hat[:, :num_classes]
            logits_ood = logits_hat[:, num_classes:]
            cs_loss = F.cross_entropy(logits_hat, targets)
            detect_loss = -(logits_ood.mean(1) - torch.logsumexp(logits_ood, dim=1)).mean()
            regularization = self.gamma_2 * torch.mean(torch.norm(id_noise, p=2, dim=1))
            adv_loss = cs_loss + detect_loss - regularization
            grads = torch.autograd.grad(adv_loss, [ood_noise])[0]
            # grads /= (grads**2).sum(-1).sqrt().unsqueeze(1) + 1e-16
            norm_grads = torch.norm(grads, p=2, dim=-1, keepdim=True) + 1e-8
            grads /= norm_grads
            grads[torch.isinf(grads) | torch.isnan(grads)] = 0
            grad_clip_value = 1.0
            grads = torch.clamp(grads, -grad_clip_value, grad_clip_value)
            ood_noise = ood_noise.detach() + self.strength * grads.detach()
            optimizer.zero_grad()

            if prev_loss is not None and abs(adv_loss.item() - prev_loss) < self.epsilon:
                break
            prev_loss = adv_loss.item()

        return id_noise.detach(), ood_noise.detach()

    def forward(self, img_features, id_features, ood_features, targets, optimizer):
        return self.forward_adl(img_features, id_features, ood_features, targets, optimizer)

    def forward_adl(self, img_features, id_features, ood_features, targets, optimizer):
        num_classes = id_features.shape[0]
        id_noise, ood_noise = self.generate_adl(
            self.num_iteration, img_features, id_features, ood_features, targets, optimizer
        )
        id_features_hat, ood_features_hat = id_features + id_noise, ood_features + ood_noise
        logits = self.logit_scale * img_features @ torch.cat([id_features_hat, ood_features_hat], dim=0).T
        id_logits = logits[:, :num_classes]
        ood_logits = logits[:, num_classes:]

        cs_loss = F.cross_entropy(logits, targets, reduce=False)

        detect_loss = -(ood_logits.mean(1) - torch.logsumexp(ood_logits, dim=1))

        r_sur1 = (id_noise.abs()).mean(-1).mean()
        r_sur2 = (ood_noise.abs()).mean(-1).mean()
        # self.gamma -= self.beta * (self.rho - (r_sur1 +r_sur2 ).detach())
        # self.gamma = self.gamma.clamp(min=0.0, max=self.gamma_init)

        self.gamma_1 -= self.beta * (self.rho - (id_noise.abs()).mean(-1).mean().detach())
        self.gamma_1 = self.gamma_1.clamp(min=0.0, max=self.gamma_init)

        self.gamma_2 -= self.beta * (self.rho - (ood_noise.abs()).mean(-1).mean().detach())
        self.gamma_2 = self.gamma_2.clamp(min=0.0, max=self.gamma_init)

        return cs_loss, detect_loss, r_sur1, r_sur2
