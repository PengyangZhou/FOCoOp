import logging
import os
import warnings

import torch
import wandb

from config import parser
from src.data.data_partition.data_partition import dirichlet_load_train_clip, pathological_load_train_clip
from src.data.data_partition.data_partition import load_domain_train
from src.utils.main_utils import make_save_path, get_server_and_client, set_seed, build_clip_model

warnings.filterwarnings("ignore")
torch.set_num_threads(8)
os.environ["HTTPS_PROXY"] = "http://10.162.243.198:7890"


def run():
    if not args.debug:
        wandb.init()
        for key in dict(wandb.config):
            setattr(args, key, dict(wandb.config)[key])
        wandb.config.update(args)
        wandb.run.name = f"{args.method}"

    set_seed(args.seed)

    save_path = make_save_path(args)
    if args.checkpoint_path == "default":
        setattr(args, "checkpoint_path", save_path)

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(save_path, "output.log"),
        format="[%(asctime)s %(levelname)s] %(message)s",
        filemode="w",
    )

    logging.info(f"-------------------- configuration --------------------")
    for key, value in args._get_kwargs():
        logging.info(f"configuration {key}: {value}")

    # ---------- dataset preprocess ----------
    print("prepare dataset...")

    if args.backbone == "clip":
        if args.id_dataset in ["office", "domainnet", "pacs"]:
            train_datasets, num_class, class_names = load_domain_train(
                args.dataset_path, args.id_dataset, args.leave_out
            )
            setattr(args, "num_client", len(train_datasets))
        else:
            if not args.pathological:
                print("Sampling by Dirichlet ...")
                train_datasets, num_class, class_names = dirichlet_load_train_clip(
                    args.dataset_path, args.id_dataset, args.num_client, args.num_shot, args.alpha, args.dataset_seed
                )
            else:
                print("Sampling by pathological ...")
                train_datasets, num_class, class_names = pathological_load_train_clip(
                    args.dataset_path,
                    args.id_dataset,
                    args.num_client,
                    args.num_shot,
                    args.class_per_client,
                    args.dataset_seed,
                    non_overlap=args.non_overlap,
                )
    args.class_names = class_names
    print("num_client:", args.num_client)

    # -------------- add for out prompts --------------
    args.ex_class_names = ["X"] * args.num_ex_prompt

    # ---------- construct backbone model ----------
    print("init server and clients...")
    if args.backbone == "clip":
        backbone = build_clip_model(args)
    else:
        raise NotImplementedError("backbone should be CLIP")

    device = torch.device(args.device)

    # ---------- construct server and clients ----------
    server_args = {
        "join_ratio": args.join_ratio,
        "checkpoint_path": args.checkpoint_path,
        "backbone": backbone,
        "device": device,
        "debug": args.debug,
        "alpha": args.alpha,
        "id_dataset": args.id_dataset,
        "num_client": args.num_client,
        "leave_out": args.leave_out,
        "pathological": args.pathological,
        "non_overlap": args.non_overlap,
        "class_per_client": args.class_per_client,
        "dataset_path": args.dataset_path,
        "num_workers": args.num_workers,
        "score_method": args.score_method,
    }
    client_args = [
        {
            "cid": cid,
            "device": device,
            "epochs": args.local_epochs,
            "backbone": backbone,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "train_id_dataset": train_datasets[cid],
            "checkpoint_path": args.checkpoint_path,
        }
        for cid in range(args.num_client)
    ]

    Server, Client, client_args, server_args = get_server_and_client(args, client_args, server_args)
    server = Server(server_args)
    clients = [Client(client_args[idx]) for idx in range(args.num_client)]
    server.clients.extend(clients)

    # ---------- fit the model ----------
    logging.info("------------------------------ fit the model ------------------------------")
    for t in range(args.communication_rounds):
        logging.info(f"------------------------- round {t} -------------------------")
        print(f"------------------------- round {t} -------------------------")
        import time
        start = time.time()
        server.fit()
        end = time.time()
        print("time: ", end - start)
    print("save the model...")
    server.make_checkpoint(args.communication_rounds)
    print("done.")
    server.test_on_round(server_args)


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running for method: {args.method}---------------")

    if args.id_dataset in ["office", "domainnet", "pacs"]:
        project_name = f"{args.id_dataset}_{args.num_client}clients"
    elif args.pathological:
        if args.non_overlap:
            project_name = f"{args.id_dataset}_{args.num_client}clients_pathological{args.class_per_client}_non_overlap"
        else:
            project_name = f"{args.id_dataset}_{args.num_client}clients_pathological{args.class_per_client}"
    else:
        project_name = f"{args.id_dataset}_{args.num_client}clients_{args.alpha}alpha"
    if args.debug:
        wandb.init(mode="disabled")
        run()
    else:
        sweep_configuration = {
            "method": "grid",
            "parameters": {
                "num_shot": {
                    "values": [8],
                },
            },
        }

        sweep_id = wandb.sweep(
            sweep_configuration,
            project=project_name,
        )
        wandb.agent(sweep_id, function=run)
