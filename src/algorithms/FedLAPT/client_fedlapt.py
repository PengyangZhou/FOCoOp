import pdb
from abc import ABC

import numpy as np
import torch
from torch.nn import functional as F

from src.algorithms.FedLAPT.class_queue import ClassTensorQueues
from src.algorithms.FedLAPT.loss import soft_cross_entropy
from src.algorithms.FedLAPT.mixup import mixup_data
from src.algorithms.base.client_base import BaseClient


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


class FedLAPTClient(BaseClient, ABC):
    def __init__(self, client_args):
        super().__init__(client_args)
        self.alpha = client_args["alpha"]
        self.beta = client_args["beta"]
        self.total_gs_num = client_args["total_gs_num"]
        self.selected_gs_num = client_args["selected_gs_num"]
        self.gs_loss_weight = client_args["gs_loss_weight"]
        self.loss_weights = client_args["loss_weights"].split("-")
        for i in range(len(self.loss_weights)):
            self.loss_weights[i] = float(self.loss_weights[i])
        self.use_gs = client_args["use_gs"]
        self.soft_split = client_args["soft_split"]

        self.mix_strategy = client_args["mix_strategy"]
        self.loss_components = client_args["loss_components"]
        self.text_features = (
            self.backbone.text_features.unsqueeze(1) if self.backbone.text_center else self.backbone.text_features
        )
        self.text_features_unselected = (
            self.backbone.text_features_unselected.unsqueeze(1)
            if self.backbone.text_center
            else self.backbone.text_features
        )
        self.extra_ood_class = True

        assert self.beta >= 0.0
        if self.beta > 2:
            print("pay attention that many mixed samples will be treaded as pure ood data with the beta", self.beta)

        self.optimizer = torch.optim.SGD(
            params=self.backbone.prompt_learner.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            # nesterov=True
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                self.epochs * len(self.train_id_dataloader),
                1,
                1e-8,
            ),
        )
        self.num_classes = client_args["n_cls"]
        self.num_ood_classes = client_args["num_ex_prompt"]
        self.queue_capacity = client_args["queue_capacity"]
        # for gaussian modeling and resampling
        self.class_query = ClassTensorQueues(class_num=self.num_classes, capacity=self.queue_capacity, feature_dim=512)
        self.iter_recomputation = client_args["iter_recomputation"]
        self.iter_count = self.iter_recomputation
        self.pre_queue = client_args["pre_queue"]
        self.binary_loss = torch.nn.BCELoss()

    def train(self):
        self.backbone.to(self.device)
        self.text_features = self.text_features.to(self.device)
        self.text_features_unselected = self.text_features_unselected.to(self.device)

        print(f"---------- training client {self.cid} ----------")
        for epoch in range(self.epochs):
            print(f"---------- epoch {epoch}  ----------")

            if self.use_gs and self.pre_queue and (not self.class_query.is_full()):
                self.backbone.eval()
                print("Filling the class queue with image features")
                while not self.class_query.is_full():
                    for classifier_set in self.train_id_dataloader:
                        if len(classifier_set[0]) == 1:
                            continue
                    data = classifier_set[0].to(self.device)
                    targets = classifier_set[1].to(self.device)
                    with torch.no_grad():
                        image_features, _, _ = self.backbone(data, return_feat=True)
                    self.class_query.push(image_features, targets)
                print("class queue prepared.")

            self.backbone.train()

            for classifier_set in self.train_id_dataloader:
                data = classifier_set[0]
                targets = classifier_set[1]

                binary_labels = (targets < self.num_classes).long().float()

                batch_size = targets.shape[0]

                soft_label = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()

                data, targets, soft_label = data.to(self.device), targets.to(self.device), soft_label.to(self.device)

                if self.num_ood_classes + self.num_classes == self.num_classes:
                    self.extra_ood_class = False
                    ood_label = soft_label.new_ones((soft_label.size(0), soft_label.size(1)))
                    ood_label = ood_label / soft_label.size(1)
                else:
                    self.extra_ood_class = True
                    ood_expand = soft_label.new_zeros((batch_size, self.num_ood_classes))
                    soft_label = torch.cat((soft_label, ood_expand), dim=1)
                    ood_label = soft_label.new_zeros((soft_label.size(0), soft_label.size(1)))
                    ood_label[:, -self.num_ood_classes :] = 1

                if self.mix_strategy == "mixup" or self.mix_strategy == "cutmix":
                    mixup_x, mixup_y_a, mixup_y_b, lam = mixup_data(
                        data, targets, soft_label, self.alpha, self.mix_strategy
                    )
                    new_x = torch.cat([data, mixup_x], dim=0)
                    image_features, text_features, logit_scale = self.backbone(new_x, return_feat=True)
                    logits = logit_scale * image_features @ text_features.t()
                    image_features = image_features[:batch_size, :]
                    cossim_id_ood = logits[:batch_size, self.num_classes :] / logit_scale
                    logits = [logits[:batch_size, :], logits[batch_size:, :]]
                    soft_labels = [soft_label]
                elif self.mix_strategy == "vanilla":
                    image_features, text_features, logit_scale = self.backbone(data, return_feat=True)
                    mixup_x, part_y_a, part_y_b, lam = mixup_data(
                        image_features, targets, soft_label, self.alpha, "manimix"
                    )
                    logits_all = logit_scale * image_features @ text_features.t()
                    cossim_id_ood = logits_all[:batch_size, self.num_classes :] / logit_scale

                    binary_prob = F.softmax(logits_all, dim=1)[:, : self.num_classes].sum(1)
                    logits = [binary_prob]
                    soft_labels = [binary_labels]

                elif self.mix_strategy == "wccm":
                    image_features, text_features, logit_scale = self.backbone(data, return_feat=True)
                    mixup_x_wccm, part_y_a, part_y_b, lam = mixup_data(
                        [image_features, self.text_features], targets, soft_label, self.alpha, self.mix_strategy
                    )
                    logits_all = logit_scale * mixup_x_wccm @ text_features.t()
                    cossim_id_ood = logits_all[:batch_size, self.num_classes :] / logit_scale
                    id_logits = F.softmax(logits_all, dim=1)[:, : self.num_classes]
                    binary_prob = id_logits.sum(1)
                    logits = [binary_prob]
                    soft_labels = [binary_labels]

                elif self.mix_strategy == "wccm_cd":
                    image_features, text_features, logit_scale = self.backbone(data, return_feat=True)
                    mixup_x_wccm, _, _, _ = mixup_data(
                        [image_features, self.text_features], targets, soft_label, self.alpha, "wccm"
                    )
                    mixup_x_cd, part_y_a, part_y_b, lam = mixup_data(
                        image_features, targets, soft_label, self.alpha, "cross_dis"
                    )

                    logits_all = mixup_x_wccm @ text_features.t()
                    logits_cd = mixup_x_cd @ text_features.t()

                    cossim_id_ood = logits_all[:batch_size, self.num_classes :] / logit_scale
                    binary_prob = F.softmax(logits_all, dim=1)[:, : self.num_classes].sum(1)
                    logits = [binary_prob]
                    soft_labels = [binary_labels]

                elif self.mix_strategy == "manimix" or self.mix_strategy == "geomix":
                    image_features, text_features, logit_scale = self.backbone(data, return_feat=True)
                    mixup_x, part_y_a, part_y_b, lam = mixup_data(
                        image_features, targets, soft_label, self.alpha, self.mix_strategy
                    )
                    new_x = torch.cat([image_features, mixup_x], dim=0)
                    logits = logit_scale * new_x @ text_features.t()
                    cossim_id_ood = logits[:batch_size, self.num_classes :] / logit_scale
                    logits = [logits[:batch_size, :], logits[batch_size:, :]]
                    soft_labels = [soft_label]
                elif self.mix_strategy == "manimixrev":
                    image_features, text_features, logit_scale = self.backbone(data, return_feat=True)
                    mixup_x, part_y_a, part_y_b, lam = mixup_data(
                        image_features, targets, soft_label, self.alpha, self.mix_strategy
                    )
                    new_x = torch.cat([image_features, mixup_x], dim=0)
                    logits = logit_scale * new_x @ text_features.t()
                    cossim_id_ood = logits[:batch_size, self.num_classes :] / logit_scale
                    logits = [logits[:batch_size, :], logits[batch_size:, :]]
                    soft_labels = [soft_label]
                elif self.mix_strategy == "manimix_wccm" or self.mix_strategy == "geomix_wccm":
                    strategy_one, strategy_two = self.mix_strategy.split("_")
                    image_features, text_features, logit_scale = self.backbone(data, return_feat=True)

                    mixup_x, part_y_a, part_y_b, lam = mixup_data(
                        image_features, targets, soft_label, self.alpha, strategy_one
                    )
                    mixup_x_wccm, _, _, _ = mixup_data(
                        [image_features, self.text_features], targets, soft_label, self.alpha, self.mix_strategy
                    )
                    new_x = torch.cat([image_features, mixup_x_wccm, mixup_x], dim=0)
                    logits = logit_scale * new_x @ text_features.t()
                    cossim_id_ood = logits[: batch_size * 2, self.num_classes :] / logit_scale  # B*C

                    logits = [
                        logits[:batch_size, :],
                        logits[batch_size : batch_size * 2, :],
                        logits[batch_size * 2 :, :],
                    ]
                    soft_labels = [soft_label, soft_label]
                elif self.mix_strategy == "mani_cccm" or self.mix_strategy == "geo_cccm":
                    image_features, text_features, logit_scale = self.backbone(data, return_feat=True)
                    mixup_x, part_y_a, part_y_b, lam = mixup_data(
                        [image_features, self.text_features], targets, soft_label, self.alpha, self.mix_strategy
                    )
                    new_x = torch.cat([image_features, mixup_x], dim=0)
                    logits = logit_scale * new_x @ text_features.t()
                    cossim_id_ood = logits[:batch_size, self.num_classes :] / logit_scale  # B*C

                    logits = [logits[:batch_size, :], logits[batch_size:, :]]
                    soft_labels = [soft_label]
                elif self.mix_strategy == "mixup_manimix_wccm" or self.mix_strategy == "mixup_geomix_wccm":
                    image_mix_strategy, strategy_one, strategy_two = self.mix_strategy.split("_")
                    wccm_strategy = strategy_one + "_" + strategy_two
                    mixup_x_image, part_y_a_image, part_y_b_image, lam_image = mixup_data(
                        data, targets, soft_label, self.alpha, image_mix_strategy
                    )

                    id_lam_image = 1 - self.beta * lam_image
                    ood_lam_image = self.beta * lam_image
                    mixup_y_image = ood_lam_image * ood_label + id_lam_image * (
                        lam_image * part_y_a_image + (1 - lam_image) * part_y_b_image
                    )

                    new_x_image = torch.cat([data, mixup_x_image], dim=0)
                    image_features_vanilla_mixed, text_features, logit_scale = self.backbone(
                        new_x_image, return_feat=True
                    )
                    image_features, mixed_image_features = torch.chunk(image_features_vanilla_mixed, 2, dim=0)

                    mixup_x, part_y_a, part_y_b, lam = mixup_data(
                        image_features, targets, soft_label, self.alpha, strategy_one
                    )
                    mixup_x_wccm, _, _, _ = mixup_data(
                        [image_features, self.text_features], targets, soft_label, self.alpha, wccm_strategy
                    )
                    new_x = torch.cat([image_features, mixup_x_wccm, mixed_image_features, mixup_x], dim=0)
                    logits = logit_scale * new_x @ text_features.t()
                    logits = [
                        logits[:batch_size, :],
                        logits[batch_size : batch_size * 2, :],
                        logits[batch_size * 2 : batch_size * 3, :],
                        logits[batch_size * 3 :, :],
                    ]
                    soft_labels = [soft_label, soft_label, mixup_y_image]
                else:
                    raise NotImplementedError

            ood_lam = self.beta * lam
            if ood_lam > 1:
                ood_lam = 1
            id_lam = 1 - ood_lam

            mixup_y = ood_lam * ood_label + id_lam * (lam * part_y_a + (1 - lam) * part_y_b)
            soft_labels.append(mixup_y)
            assert len(soft_labels) >= len(logits)

            if self.loss_components == "binaryce":
                loss = self.binary_loss(logits[0].float(), soft_labels[0]) * self.loss_weights[0]
            elif self.loss_components == "entropy":
                loss = -(
                    logits[0].float() * torch.log(logits[0].float() + 1e-6)
                    + (1 - logits[0].float()) * torch.log(1 - logits[0].float() + 1e-6)
                ).mean()
            elif self.loss_components == "multice":

                loss = soft_cross_entropy(logits_all, soft_label) * self.loss_weights[0]
                if self.mix_strategy == "wccm_cd":
                    loss += soft_cross_entropy(logits_cd, mixup_y) * self.loss_weights[1]

            elif self.loss_components == "textentropy":

                batch_size = image_features.size()[0]
                label_num, text_num = self.text_features_unselected.size()[0], self.text_features_unselected.size()[1]
                random_indices = torch.randint(0, text_num, (batch_size,)).cuda()
                label_indices = torch.randint(0, label_num, (batch_size,)).cuda()

                sampled_unselected_text_feat = self.text_features_unselected[label_indices, random_indices, :]
                logits_unseltext = logit_scale * sampled_unselected_text_feat @ text_features.t()
                binary_prob_unseltext = F.softmax(logits_unseltext, dim=1)[:, : self.num_classes].sum(1)
                loss = -(
                    binary_prob_unseltext.float() * torch.log(binary_prob_unseltext.float() + 1e-6)
                    + (1 - binary_prob_unseltext.float()) * torch.log(1 - binary_prob_unseltext.float() + 1e-6)
                ).mean()

            elif self.loss_components == "textmultice":
                batch_size = image_features.size()[0]
                label_num, text_num = self.text_features.size()[0], self.text_features.size()[1]
                random_indices = torch.randint(0, text_num, (batch_size,)).cuda()
                label_indices = torch.randint(0, label_num, (batch_size,)).cuda()
                sampled_text_feat = self.text_features[label_indices, random_indices, :]
                logits_text = logit_scale * sampled_text_feat @ text_features.t()
                Y_one_hot = F.one_hot(label_indices, num_classes=label_num)
                loss = soft_cross_entropy(logits_text, Y_one_hot) * self.loss_weights[0]
            else:
                raise NotImplementedError
            if torch.isnan(text_features).any():
                pdb.set_trace()

            if "abscos" in self.loss_components:

                max_ood_cossim, _ = torch.max(cossim_id_ood, dim=1)
                binary_loss = -torch.log(1 - torch.sigmoid(max_ood_cossim)).mean()
                loss += binary_loss * self.loss_weights[len(soft_labels)]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        self.backbone.cpu()
        return {
            "backbone": {
                "prompt_learner.ctx": self.backbone.prompt_learner.ctx.detach().clone(),
                "prompt_learner.ctx_ood": self.backbone.prompt_learner.ctx_ood.detach().clone(),
            },
            "acc": 0.0,
        }
