from abc import ABC

import torch
from sklearn.metrics import accuracy_score
from torch.nn import functional as F

from src.algorithms.base.client_base import BaseClient

cos = torch.nn.CosineSimilarity(dim=-1)


class FedPGPClient(BaseClient, ABC):
    def __init__(self, client_args):
        super().__init__(client_args)

        self.temp = client_args["temp"]
        self.mu = client_args["mu"]
        self.optimizer = torch.optim.SGD(
            params=self.backbone.prompt_learner.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

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

                text_features_0, text_features_sigma, text_features_UV, text_features, output = self.backbone(data)
                posi = cos(text_features_0, text_features_sigma)
                nega = cos(text_features_sigma, text_features)

                logits = torch.cat((posi.reshape(-1, 1), nega.reshape(-1, 1)), dim=1)
                logits /= self.temp
                targets2 = torch.zeros(logits.size(0)).to(self.device).long()

                pred = output.data.max(1)[1]
                accuracy.append(accuracy_score(list(targets.data.cpu().numpy()), list(pred.data.cpu().numpy())))

                self.optimizer.zero_grad()

                loss2 = F.cross_entropy(logits, targets2)
                loss = F.cross_entropy(output, targets)
                loss += self.mu * loss2

                loss.backward()
                self.optimizer.step()

        self.backbone.cpu()
        return {
            "backbone": {"prompt_learner.sigma": self.backbone.prompt_learner.sigma.detach().clone()},
            "acc": sum(accuracy) / len(accuracy),
        }
