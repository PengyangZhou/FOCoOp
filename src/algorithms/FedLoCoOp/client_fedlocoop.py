from abc import ABC

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from src.algorithms.FedLoCoOp.loss import entropy_select_topk
from src.algorithms.base.client_base import BaseClient
from src.utils.accuracy import compute_fnr, compute_auroc


class FedLoCoOpClient(BaseClient, ABC):
    def __init__(self, client_args):
        super().__init__(client_args)
        self.num_classes = client_args["n_cls"]

        for name, param in self.backbone.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.optimizer = torch.optim.SGD(
            params=self.backbone.prompt_learner.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.precision = client_args["precision"]
        self.lambda_value = client_args["lambda_local"]
        self.top_k = client_args["top_k"]

        self.scaler = GradScaler() if self.precision == "amp" else None

    def train(self):
        self.backbone.to(self.device)

        accuracy = []
        print(f"---------- training client {self.cid} ----------")
        for epoch in range(self.epochs):
            print(f"---------- epoch {epoch}  ----------")
            self.backbone.train()
            for classifier_set in self.train_id_dataloader:
                if len(classifier_set[0]) == 1:
                    continue
                data = classifier_set[0].to(self.device)
                targets = classifier_set[1].to(self.device)

                if self.precision == "amp":
                    with autocast():
                        output, output_local = self.backbone(data)
                        pred = output.data.max(1)[1]
                        accuracy.append(accuracy_score(list(targets.data.cpu().numpy()), list(pred.data.cpu().numpy())))

                        loss_id = F.cross_entropy(output, targets)

                        batch_size, num_of_local_feature = output_local.shape[0], output_local.shape[1]
                        output_local = output_local.view(batch_size * num_of_local_feature, -1)
                        loss_en = -entropy_select_topk(output_local, self.top_k, targets, num_of_local_feature)

                        loss = loss_id + self.lambda_value * loss_en

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    output, output_local = self.backbone(data)
                    pred = output.data.max(1)[1]
                    accuracy.append(accuracy_score(list(targets.data.cpu().numpy()), list(pred.data.cpu().numpy())))

                    loss_id = F.cross_entropy(output, targets)

                    batch_size, num_of_local_feature = output_local.shape[0], output_local.shape[1]
                    output_local = output_local.view(batch_size * num_of_local_feature, -1)
                    loss_en = -entropy_select_topk(output_local, self.top_k, targets, num_of_local_feature)

                    loss = loss_id + self.lambda_value * loss_en

                    loss.backward()
                    self.optimizer.step()

        self.backbone.cpu()
        return {
            "backbone": {"prompt_learner.ctx": self.backbone.prompt_learner.ctx.detach().clone()},
            "acc": sum(accuracy) / len(accuracy),
        }

    @torch.no_grad()
    def test_classification_detection_ability(self, in_loader, ood_loader, score_method="msp"):
        self.backbone.to(self.device)
        self.backbone.eval()

        ood_score_id = []
        ood_score_ood = []
        id_accuracy = []
        for data, targets in in_loader:
            if len(data) == 1:
                continue
            data, targets = data.to(self.device), targets.to(self.device)
            logit, _ = self.backbone.forward(data)
            pred = logit.data.max(1)[1]
            id_accuracy.append(accuracy_score(list(targets.data.cpu().numpy()), list(pred.data.cpu().numpy())))
            if score_method == "energy":
                ood_score_id.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))
            elif score_method == "msp":
                ood_score_id.extend(list(np.max(F.softmax(logit, dim=1).cpu().numpy(), axis=1)))

        for data, _ in ood_loader:
            if len(data) == 1:
                continue
            data = data.to(self.device)
            logit, _ = self.backbone.forward(data)
            if score_method == "energy":
                ood_score_ood.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).data.cpu().numpy()))
            elif score_method == "msp":
                ood_score_ood.extend(list(np.max(F.softmax(logit, dim=1).cpu().numpy(), axis=1)))

        id_accuracy = sum(id_accuracy) / len(id_accuracy)
        if score_method == "energy":
            fpr95 = compute_fnr(np.array(ood_score_ood), np.array(ood_score_id))
            auroc = compute_auroc(np.array(ood_score_ood), np.array(ood_score_id))
        elif score_method == "msp":
            fpr95 = compute_fnr(np.array(ood_score_id), np.array(ood_score_ood))
            auroc = compute_auroc(np.array(ood_score_id), np.array(ood_score_ood))

        self.backbone.cpu()
        return id_accuracy, fpr95, auroc

    @torch.no_grad()
    def test_corrupt_accuracy(self, cor_loader):
        self.backbone.to(self.device)
        self.backbone.eval()

        accuracy = []
        for data, targets in cor_loader:
            if len(data) == 1:
                continue
            data, targets = data.to(self.device), targets.to(self.device)
            logit, _ = self.backbone(data)
            pred = logit.data.max(1)[1]
            accuracy.append(accuracy_score(list(targets.data.cpu().numpy()), list(pred.data.cpu().numpy())))
        return sum(accuracy) / len(accuracy)
