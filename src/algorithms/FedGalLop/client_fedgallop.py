from abc import ABC
from typing import Optional

import torch
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.cuda.amp import autocast

from src.algorithms.FedGalLop.loss import GlobalLocalLoss
from src.algorithms.FedGalLop.vlp_tools import topk_reduce
from src.algorithms.base.client_base import BaseClient
from src.models.clip_w_local.modified_clip_model_locoop import get_params_group


class FedGalLopClient(BaseClient, ABC):
    def __init__(self, client_args):
        super().__init__(client_args)
        self.num_classes = client_args["n_cls"]

        params_group = get_params_group(self.backbone)

        self.optimizer = torch.optim.SGD(
            params=params_group,
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self.precision = client_args["precision"]
        self.lambda_value = client_args["lambda_local"]
        self.use_global_loss = client_args["use_global_loss"]
        self.use_local_loss = client_args["use_local_loss"]
        self.global_dropout_p = client_args["global_dropout_p"]

        self.top_k = [5, 10, 15, 20]

        self.loss_fn = GlobalLocalLoss(
            use_global_loss=self.use_global_loss,
            use_local_loss=self.use_local_loss,
            topk=self.top_k,
            global_dropout_p=self.global_dropout_p,
        )

    def train(self):
        self.backbone.to(self.device)
        if not self.backbone.learn_global_prompt and not self.backbone.learn_local_prompt:
            with torch.no_grad(), autocast(enabled=self.precision == "amp"):
                text_features, local_text_features = self.backbone.get_text_features()
                text_features /= text_features.norm(dim=-1, keepdim=True)
                local_text_features /= local_text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = local_text_features = None
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

                global_logits, local_logits = self.backbone(
                    data, text_features, local_text_features, return_logits=True
                )
                loss = self.loss_fn(global_logits, local_logits, targets, self.backbone.logit_scale.exp())

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                gl_probs, global_probs, local_probs = self.create_prediction_scores(global_logits, local_logits)
                pred = gl_probs.data.max(1)[1]
                accuracy.append(accuracy_score(list(targets.data.cpu().numpy()), list(pred.data.cpu().numpy())))

        self.backbone.cpu()
        return {
            "backbone": {"global_prompt": self.backbone.global_prompt.detach().clone()},
            "acc": sum(accuracy) / len(accuracy),
        }

    @torch.no_grad()
    def create_prediction_scores(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> Tensor:
        logit_scale = self.backbone.logit_scale.exp()
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(logit_scale * global_logits, dim=-1)

        if local_logits is None:
            local_probs = None
            gl_probs = global_probs
        else:
            local_logits = topk_reduce(local_logits, topk=self.top_k)
            local_logits = local_logits.mean(dim=-1)
            local_probs = torch.softmax(logit_scale * local_logits, dim=-1)
            gl_logits = (global_logits + local_logits) / 2
            gl_probs = torch.softmax(logit_scale * gl_logits, dim=-1)

        return gl_probs, global_probs, local_probs

    def make_checkpoint(self):
        checkpoint = {
            "backbone": {
                "global_prompt": self.backbone.global_prompt.detach().clone(),
                "local_prompts": self.backbone.local_prompts.detach().clone(),
                "local_proj.linear.weight": self.backbone.local_proj.linear.weight.detach().clone(),
            },
        }

        return checkpoint

    def load_checkpoint(self, checkpoint):
        if "backbone" in checkpoint:
            self.backbone.load_state_dict(checkpoint["backbone"], strict=False)

        return checkpoint
