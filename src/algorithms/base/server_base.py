import copy
import os
import random
from abc import abstractmethod

import torch
import wandb


class BaseServer:
    def __init__(self, server_args):
        self.clients = []
        self.join_raio = server_args["join_ratio"]
        self.checkpoint_path = server_args["checkpoint_path"]
        self.backbone = copy.deepcopy(server_args["backbone"])
        self.device = server_args["device"]
        self.debug = server_args["debug"]
        self.num_workers = server_args["num_workers"]
        self.test_batch_size = 256

    def select_clients(self):
        if self.join_raio == 1.0:
            return self.clients
        else:
            return random.sample(self.clients, int(round(len(self.clients) * self.join_raio)))

    @abstractmethod
    def fit(self):
        pass

    @staticmethod
    def model_average(client_net_states, client_weights):
        state_avg = copy.deepcopy(client_net_states[0])
        client_weights = [w / sum(client_weights) for w in client_weights]

        for k in state_avg.keys():
            state_avg[k] = torch.zeros_like(state_avg[k])
            for i, w in enumerate(client_weights):
                state_avg[k] = state_avg[k] + client_net_states[i][k] * w

        return state_avg

    def make_checkpoint(self, current_round, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_path, f"model_{current_round}.pt")
        else:
            checkpoint_path = os.path.join(self.checkpoint_path, checkpoint_path)

        assert self.global_net_state is not None, "The global model is not initialized!"
        checkpoint = {
            "server": self.global_net_state,
            "clients": [client.make_checkpoint() for client in self.clients],
        }
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint):
        self.backbone.load_state_dict(checkpoint["server"], strict=False)
        for client, checkpoint in zip(self.clients, checkpoint["clients"]):
            client.load_checkpoint(checkpoint)

    def test_classification_detection_ability(self, client_id_loaders, ood_loader, score_method="msp"):
        auroc = 0.0
        fpr95 = 0.0
        accuracy = 0.0

        test_samples = [len(id_loader) for id_loader in client_id_loaders]
        client_weights = [x / sum(test_samples) for x in test_samples]

        for client, id_loader, w in zip(self.clients, client_id_loaders, client_weights):
            print("test_classification_detection_ability:", client.cid)
            client_accuracy, client_fpr95, client_auroc = client.test_classification_detection_ability(
                id_loader, ood_loader, score_method=score_method
            )
            accuracy += client_accuracy * w
            fpr95 += client_fpr95 / len(self.clients)
            auroc += client_auroc / len(self.clients)

        return accuracy, fpr95, auroc

    def test_corrupt_accuracy(self, client_cor_loaders):
        cor_accuracy = {}
        for cor_type, cor_loaders in client_cor_loaders.items():
            cor_accuracy[cor_type] = 0.0
            test_samples = [len(cor_loader) for cor_loader in cor_loaders]
            client_weights = [x / sum(test_samples) for x in test_samples]

            for client, cor_loader, w in zip(self.clients, cor_loaders, client_weights):
                cor_accuracy[cor_type] += client.test_corrupt_accuracy(cor_loader) * w

        return cor_accuracy

    def test_on_round(self, server_args):
        from src.data.data_partition.data_partition import pathological_load_test_clip, dirichlet_load_test_clip
        from src.data.data_partition.data_partition import load_test_ood_clip, load_domain_test
        from torch.utils.data import DataLoader

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
        client_id_loaders = [
            DataLoader(dataset=id_datasets[idx], batch_size=self.test_batch_size, shuffle=False, num_workers=2)
            for idx in range(server_args["num_client"])
        ]
        ood_loader = DataLoader(dataset=ood_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=2)

        if server_args["id_dataset"] in ["cifar10", "cifar100", "tinyimagenet"]:
            cor_loaders = dict()
            for cor_type in corrupt_list:
                cor_loaders[cor_type] = [
                    DataLoader(
                        dataset=cor_datasets[idx][cor_type],
                        batch_size=self.test_batch_size,
                        shuffle=False,
                        num_workers=2,
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
        elif server_args["id_dataset"] in ["food101", "caltech101", "flowers102", "pet37", "dtd"]:
            cor_accuracy = None
            wandb.log(
                {"id_accuracy": id_accuracy, "fpr95": fpr95, "auroc": auroc},
            )
        else:
            cor_accuracy = None
            wandb.log({"id_accuracy": id_accuracy})

        print(server_args["id_dataset"], server_args["alpha"])
        print(f"id_accuracy: {id_accuracy}, fpr95: {fpr95}, auroc: {auroc}, cor_accuracy: {cor_accuracy}")
