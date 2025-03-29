import logging
import time

import wandb

from src.algorithms.FedGalLop.client_fedgallop import FedGalLopClient
from src.algorithms.base.server_base import BaseServer


class FedGalLopServer(BaseServer):
    def __init__(self, server_args):
        super().__init__(server_args)
        self.global_net_state = None

    def fit(self):
        client_net_states = []
        client_train_time = []
        client_accuracy = []
        client_weights = []
        active_clients = self.select_clients()
        for client in active_clients:
            client_weights.append(len(client.train_id_dataloader))
            client: FedGalLopClient
            start_time = time.time()
            report = client.train()
            end_time = time.time()
            client_net_states.append(report["backbone"])
            client_accuracy.append(report["acc"])
            logging.info(f"client{client.cid} training time: {end_time - start_time}")
            client_train_time.append(end_time - start_time)

        print(f"average client train time: {sum(client_train_time) / len(client_train_time)}")
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

        global_net_state = self.model_average(client_net_states, client_weights)
        self.global_net_state = global_net_state
        for client in self.clients:
            client.backbone.load_state_dict(global_net_state, strict=False)
        self.backbone.load_state_dict(global_net_state, strict=False)
