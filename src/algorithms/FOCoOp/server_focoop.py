import logging
import time

import ot
import torch
import wandb

from src.algorithms.FOCoOp.assignment import kmeans_pytorch
from src.algorithms.base.server_base import BaseServer


class FOCoOpServer(BaseServer):
    def __init__(self, server_args):
        super().__init__(server_args)
        self.global_net_state = None
        self.ema = server_args["ema"]
        self.num_ex_prompt = server_args["num_ex_prompt"]
        self.global_round = 0

        self.uot = server_args["uot"]

    def fit(self):
        client_net_states = []
        client_train_time = []
        client_accuracy = []
        client_weights = []
        active_clients = self.select_clients()

        for client in active_clients:
            client_weights.append(len(client.train_id_dataloader))
            client: FOCoOpServer
            start_time = time.time()
            report = client.train()
            end_time = time.time()
            client_net_states.append(report["backbone"])
            client_accuracy.append(report["acc"])
            logging.info(f"client{client.cid} training time: {end_time - start_time}")
            client_train_time.append(end_time - start_time)

        print(
            f"average client train time: {sum(client_train_time) / len(client_train_time) if len(client_train_time) != 0 else 1}"
        )
        print(
            f"average client accuracy: {sum([client_accuracy[i] * client_weights[i] / sum(client_weights) for i in range(len(active_clients))])}"
        )

        if not self.debug:
            wandb.log(
                {
                    "accuracy": sum(
                        [
                            client_accuracy[i] * client_weights[i] / sum(client_weights)
                            for i in range(len(active_clients))
                        ]
                    )
                }
            )
        aggregate_state = self.model_average(client_net_states, client_weights)

        if self.uot:
            ######### FOR UOT AGGREGATION ###########
            # apply ot alignment for Image prompts and OOD prompts
            # id_prompts=[]
            ood_prompts = []
            for prompts in client_net_states:
                ood_prompts.append(prompts["prompt_learner.ctx_ood"])

            id_prompts = aggregate_state["prompt_learner.ctx_global"]
            id_prompts = id_prompts.to(self.device)
            N, B, D = id_prompts.size()
            id_prompts_reshape = id_prompts.reshape(N, -1)

            ood_prompts = torch.cat(ood_prompts).to(self.device)
            ood_prompts_reshape = ood_prompts.reshape(ood_prompts.shape[0], -1)

            M = ot.dist(ood_prompts_reshape, id_prompts_reshape)
            M /= M.max()
            m = M.shape[0]
            n = M.shape[1]
            p_x = torch.ones(m).to(self.device) / m
            p_y = torch.ones(n).to(self.device) / n
            UOT_Pi = ot.unbalanced.mm_unbalanced(p_x, p_y, M, reg_m=0.1, div="kl")
            id_prob = UOT_Pi.sum(dim=1, keepdim=True)
            cluster_assignments, centers = kmeans_pytorch(id_prob, num_clusters=2)

            sorted_indices = torch.argsort(centers.squeeze())
            new_labels_map = {old: new for new, old in enumerate(sorted_indices.tolist())}
            adjusted_labels = torch.tensor([new_labels_map[label.item()] for label in cluster_assignments])

            n_found_id = (adjusted_labels == 1).sum()

            if n_found_id != 0:
                filter_PI = UOT_Pi[adjusted_labels == 1]
                filter_ood_prompts = ood_prompts[adjusted_labels == 1]
                filter_ood_prompts = filter_ood_prompts.reshape(filter_ood_prompts.shape[0], -1).float()
                id_prompts_new = self.ema * id_prompts + (1 - self.ema) * (filter_PI.T @ filter_ood_prompts).reshape(
                    N, B, D
                )
                aggregate_state["prompt_learner.ctx_global"] = id_prompts_new.half()

            id_prob_min, in_prob_indixes = torch.topk(id_prob.squeeze(), self.num_ex_prompt, largest=False)
            n_remaining_ood = (adjusted_labels == 0).sum()
            remaining_ood_prompts = ood_prompts[adjusted_labels == 0]
            if n_remaining_ood >= self.num_ex_prompt:
                aggregate_state["prompt_learner.ctx_ood"] = ood_prompts[in_prob_indixes]
            else:
                if n_remaining_ood != 0:
                    repeats = self.num_ex_prompt // n_remaining_ood
                    remaining = self.num_ex_prompt % n_remaining_ood
                    if repeats > 0:
                        expanded_ood_prompts = remaining_ood_prompts.repeat(repeats, 1, 1)
                    else:
                        expanded_ood_prompts = remaining_ood_prompts
                    expanded_ood_prompts = torch.cat([expanded_ood_prompts, remaining_ood_prompts[:remaining]], dim=0)

                    aggregate_state["prompt_learner.ctx_ood"] = expanded_ood_prompts.to(id_prompts)

        self.global_net_state = aggregate_state
        for client in self.clients:
            client.backbone.load_state_dict(self.global_net_state, strict=False)
        self.backbone.load_state_dict(self.global_net_state, strict=False)

        self.global_round += 1

    def test_classification_detection_ability(self, client_id_loaders, ood_loader, score_method="msp"):
        auroc = 0.0
        fpr95 = 0.0
        accuracy = 0.0
        fpr95_msp = 0.0
        fpr95_neglabel = 0.0
        auroc_msp = 0.0
        auroc_neglabel = 0.0

        test_samples = [len(id_loader) for id_loader in client_id_loaders]
        client_weights = [x / sum(test_samples) for x in test_samples]

        for client, id_loader, w in zip(self.clients, client_id_loaders, client_weights):
            print("test_classification_detection_ability:", client.cid)
            if score_method is not "msp_neglabel":
                client_accuracy, client_fpr95, client_auroc = client.test_classification_detection_ability(
                    id_loader, ood_loader, score_method=score_method
                )
                fpr95 += client_fpr95 / len(self.clients)
                auroc += client_auroc / len(self.clients)
            else:
                client_accuracy, client_fpr95_msp, client_auroc_msp, client_fpr95_neglabel, client_auroc_neglabel = (
                    client.test_classification_detection_ability(id_loader, ood_loader, score_method=score_method)
                )
                fpr95_msp += client_fpr95_msp / len(self.clients)
                auroc_msp += client_auroc_msp / len(self.clients)
                fpr95_neglabel += client_fpr95_neglabel / len(self.clients)
                auroc_neglabel += client_auroc_neglabel / len(self.clients)
            accuracy += client_accuracy * w

        if score_method is not "msp_neglabel":
            return accuracy, fpr95, auroc
        else:
            return accuracy, fpr95_msp, auroc_msp, fpr95_neglabel, auroc_neglabel

    def test_on_round(self, server_args, fraction=0.1):
        from src.data.data_partition.data_partition import pathological_load_test_clip, dirichlet_load_test_clip
        from src.data.data_partition.data_partition import load_test_ood_clip, load_domain_test
        from torch.utils.data import DataLoader

        import numpy as np
        from torch.utils.data import Subset

        def create_subset(dataset, seed=42):
            if fraction == 1:
                return dataset

            np.random.seed(seed)

            if isinstance(dataset, Subset):
                targets = np.array(dataset.dataset.targets)[dataset.indices]
                original_indices = dataset.indices
            else:
                targets = np.array(dataset.targets)
                original_indices = np.arange(len(targets))

            unique_classes = np.unique(targets)
            subset_indices = []

            for cls in unique_classes:
                cls_indices = np.where(targets == cls)[0]
                subset_size = max(1, int(len(cls_indices) * fraction))
                if subset_size > len(cls_indices):
                    subset_size = len(cls_indices)
                chosen_indices = np.random.choice(cls_indices, subset_size, replace=False)
                subset_indices.extend(original_indices[chosen_indices])

            return Subset(dataset.dataset if isinstance(dataset, Subset) else dataset, subset_indices)

        if server_args["id_dataset"] in ["office", "domainnet", "pacs"]:
            ood_dataset = load_test_ood_clip(server_args["dataset_path"], "Texture", 21, partial=True)
        else:
            ood_dataset = load_test_ood_clip(server_args["dataset_path"], "Texture", 21, partial=False)

        if server_args["id_dataset"] in ["office", "domainnet", "pacs"]:
            id_datasets, num_class, class_names = load_domain_test(
                server_args["dataset_path"], server_args["id_dataset"], server_args["leave_out"]
            )
        else:
            corrupt_list = ["brightness"]

            if server_args["pathological"]:
                id_datasets, cor_datasets, num_class, class_names = pathological_load_test_clip(
                    server_args["dataset_path"],
                    id_dataset=server_args["id_dataset"],
                    num_client=server_args["num_client"],
                    class_per_client=server_args["class_per_client"],
                    seed=21,
                    corrupt_list=corrupt_list,
                    non_overlap=server_args["non_overlap"],
                )
            else:
                id_datasets, cor_datasets, num_class, class_names = dirichlet_load_test_clip(
                    server_args["dataset_path"],
                    id_dataset=server_args["id_dataset"],
                    num_client=server_args["num_client"],
                    alpha=server_args["alpha"],
                    seed=21,
                    corrupt_list=corrupt_list,
                )

        id_datasets = [create_subset(dataset) for dataset in id_datasets]
        client_id_loaders = [
            DataLoader(dataset=id_datasets[idx], batch_size=64, shuffle=True, num_workers=2)
            for idx in range(server_args["num_client"])
        ]
        ood_loader = DataLoader(dataset=create_subset(ood_dataset), batch_size=64, shuffle=True, num_workers=2)

        if server_args["id_dataset"] in ["cifar10", "cifar100", "tinyimagenet"]:
            cor_loaders = dict()
            for cor_type in corrupt_list:
                cor_loaders[cor_type] = [
                    DataLoader(
                        dataset=create_subset(cor_datasets[idx][cor_type]),
                        batch_size=64,
                        shuffle=True,
                        num_workers=self.num_workers,
                    )
                    for idx in range(server_args["num_client"])
                ]
        id_accuracy, fpr95, auroc = self.test_classification_detection_ability(
            client_id_loaders, ood_loader, server_args["score_method"]
        )
        if server_args["id_dataset"] in ["cifar10", "cifar100", "tinyimagenet"]:
            cor_accuracy = self.test_corrupt_accuracy(cor_loaders)
            wandb.log(
                {"id_accuracy": id_accuracy, "cor_accuracy": cor_accuracy, "fpr95": fpr95, "auroc": auroc},
            )
            print(f"id_accuracy: {id_accuracy}, fpr95: {fpr95}, auroc: {auroc}")
        else:
            wandb.log({"id_accuracy": id_accuracy})
            print(f"id_accuracy: {id_accuracy}, fpr95: {fpr95}, auroc: {auroc}")
