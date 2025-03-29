from abc import ABC

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.nn import functional as F

from src.algorithms.FOCoOp.loss import BiDRO
from src.algorithms.base.client_base import BaseClient
from src.utils.accuracy import compute_fnr, compute_auroc


class FOCoOpClient(BaseClient, ABC):
    def __init__(self, client_args):
        super().__init__(client_args)

        self.num_prompt = client_args["num_prompt"]
        self.num_classes = client_args["n_cls"]
        self.num_ex_prompt = client_args["num_ex_prompt"]

        self.optimizer = torch.optim.SGD(self.backbone.prompt_learner.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, float(50))

        self.bidro_loss = BiDRO(
            self.backbone.logit_scale,
            num_iteration=4,
            num_ex_prompt=self.num_ex_prompt,
            gamma_1=client_args["gamma_1"],
            gamma_2=client_args["gamma_2"],
        )
        self.tau = client_args["tau"]
        self.dro = client_args["dro"]

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

                logits, id_local_features, ood_features, image_features = self.backbone(data, return_features=True)

                if not self.dro:
                    loss = F.cross_entropy(logits, targets)
                else:
                    loss1, loss2, r_sur1, r_sur2 = self.bidro_loss(
                        image_features, id_local_features, ood_features, targets, self.optimizer
                    )
                    loss = loss1 + 0.5 * loss2 - self.bidro_loss.gamma_1 * r_sur1 - self.bidro_loss.gamma_2 * r_sur2
                    loss = (
                        self.tau
                        * self.bidro_loss.gamma_2
                        * torch.logsumexp(loss / (self.tau * self.bidro_loss.gamma_2), dim=0)
                    )

                pred1 = logits.data.max(1)[1]

                accuracy.append(accuracy_score(list(targets.data.cpu().numpy()), list(pred1.data.cpu().numpy())))

                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
        self.backbone.cpu()

        return {
            "backbone": {
                "prompt_learner.ctx_global": self.backbone.prompt_learner.ctx_global.detach().clone(),
                "prompt_learner.ctx_ood": self.backbone.prompt_learner.ctx_ood.detach().clone(),
            },
            "acc": sum(accuracy) / len(accuracy) if len(accuracy) > 0 else 0,
        }

    @torch.no_grad()
    def test_classification_detection_ability(self, in_loader, ood_loader, score_method="neglabel"):
        self.backbone.to(self.device)
        self.backbone.eval()

        ood_score_id = []
        ood_score_ood = []
        id_accuracy = []
        for data, targets in in_loader:
            if len(data) == 1:
                continue
            data, targets = data.to(self.device), targets.to(self.device)
            logit = self.backbone.forward(data)
            id_logit, ood_logit = logit[:, : self.num_classes], logit[:, self.num_classes :]
            pred = logit[:, : self.num_classes].data.max(1)[1]
            id_accuracy.append(accuracy_score(list(targets.data.cpu().numpy()), list(pred.data.cpu().numpy())))
            if score_method == "energy":
                ood_score_id.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).cpu().numpy()))
            elif score_method == "msp":
                ood_score_id.extend(list(np.max(F.softmax(logit, dim=1).cpu().numpy(), axis=1)))
            elif score_method == "neglabel":
                ood_score_id.extend(list((id_logit.sum(dim=1) / logit.sum(dim=1)).cpu().numpy()))

        for data, _ in ood_loader:
            if len(data) == 1:
                continue
            data = data.to(self.device)
            logit = self.backbone.forward(data)
            id_logit, ood_logit = logit[:, : self.num_classes], logit[:, self.num_classes :]

            if score_method == "energy":
                ood_score_ood.extend(list(-(1.0 * torch.logsumexp(logit / 1.0, dim=1)).cpu().numpy()))
            elif score_method == "msp":
                ood_score_ood.extend(list(np.max(F.softmax(logit, dim=1).cpu().numpy(), axis=1)))
            elif score_method == "neglabel":
                ood_score_ood.extend(list((id_logit.sum(dim=1) / logit.sum(dim=1)).cpu().numpy()))

        id_accuracy = sum(id_accuracy) / len(id_accuracy)
        if score_method == "energy":
            fpr95 = compute_fnr(np.array(ood_score_ood), np.array(ood_score_id))
            auroc = compute_auroc(np.array(ood_score_ood), np.array(ood_score_id))
        elif score_method == "msp":
            fpr95 = compute_fnr(np.array(ood_score_id), np.array(ood_score_ood))
            auroc = compute_auroc(np.array(ood_score_id), np.array(ood_score_ood))
        elif score_method == "neglabel":
            fpr95 = compute_fnr(np.array(ood_score_id), np.array(ood_score_ood))
            auroc = compute_auroc(np.array(ood_score_id), np.array(ood_score_ood))

        self.backbone.cpu()

        return id_accuracy, fpr95, auroc
