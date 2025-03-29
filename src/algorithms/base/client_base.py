import copy
from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src.utils.accuracy import compute_fnr, compute_auroc


class BaseClient:

    def __init__(self, client_args):
        # ------ basic configuration ------
        self.cid = client_args["cid"]
        self.device = client_args["device"]
        self.epochs = client_args["epochs"]
        self.backbone = copy.deepcopy(client_args["backbone"])
        self.learning_rate = client_args["learning_rate"]
        self.momentum = client_args["momentum"]
        self.weight_decay = client_args["weight_decay"]
        # ------ refer to generate and iterate dataloader ------
        self.batch_size = client_args["batch_size"]
        self.num_workers = client_args["num_workers"]
        self.pin_memory = client_args["pin_memory"]
        # ------ refer to  dataloader generating ------
        self.train_id_dataset = client_args["train_id_dataset"]
        self.train_id_dataloader = DataLoader(
            dataset=self.train_id_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @abstractmethod
    def train(self):
        pass

    ############## For Local Testing ##############
    @torch.no_grad()
    def test_corrupt_accuracy(self, cor_loader):
        self.backbone.to(self.device)
        self.backbone.eval()

        accuracy = []
        for data, targets in cor_loader:
            if len(data) == 1:
                continue
            data, targets = data.to(self.device), targets.to(self.device)
            logit = self.backbone(data)
            pred = logit.data.max(1)[1]
            accuracy.append(accuracy_score(list(targets.data.cpu().numpy()), list(pred.data.cpu().numpy())))

        self.backbone.cpu()
        return sum(accuracy) / len(accuracy)

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
            logit = self.backbone.forward(data)
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
            logit = self.backbone.forward(data)
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

    ############## For Checkpoints ##############

    def make_checkpoint(self):
        checkpoint = {
            "backbone": self.backbone.prompt_learner.state_dict(),
        }

        return checkpoint

    def load_checkpoint(self, checkpoint):
        if "backbone" in checkpoint:
            self.backbone.prompt_learner.load_state_dict(checkpoint["backbone"])

        return checkpoint
