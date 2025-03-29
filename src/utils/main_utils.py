import os
import random

import numpy as np
import torch
import copy

from src.utils.clip_utils import load_clip_to_cpu


def build_clip_model(args):
    if args.id_dataset in ["food101", "caltech101", "flowers102", "pet37"]:
        clip_model, preprocess = load_clip_to_cpu("RN50", method=args.method)
        print(f"Loading CLIP (backbone: RN50)")
    else:
        clip_model, preprocess = load_clip_to_cpu("ViT-B/16", method=args.method)
        print(f"Loading CLIP (backbone: vit_b16)")

    print("Building custom CLIP")
    if args.method == "PromptFL":
        from src.models.clip_promptfl import PromptFLCLIP

        prompt_args = {
            "ctx_init": args.ctx_init,
            "class_specific_context": args.csc,
            "num_ctx": args.num_ctx,
            "class_token_position": args.class_token_position,
            "class_names": args.class_names,
        }
        backbone = PromptFLCLIP(prompt_args, prompt_args["class_names"], clip_model)
    elif args.method == "PromptFolio":
        from src.models.clip_promptfolio import PromptFolioCLIP

        args.num_prompt = 2
        prompt_args = {
            "ctx_init": args.ctx_init,
            "class_specific_context": args.csc,
            "num_ctx": args.num_ctx,
            "class_token_position": args.class_token_position,
            "class_names": args.class_names,
            "num_prompt": args.num_prompt,
            "frac": args.frac,
        }
        backbone = PromptFolioCLIP(prompt_args, prompt_args["class_names"], clip_model)
    elif args.method == "FedPGP":
        from src.models.clip_fedpgp import FedPGPCLIP

        prompt_args = {
            "ctx_init": args.ctx_init,
            "class_specific_context": args.csc,
            "num_ctx": args.num_ctx,
            "class_token_position": args.class_token_position,
            "class_names": args.class_names,
            "num_prompt": args.num_prompt,
            "bottleneck": args.bottleneck,
        }
        backbone = FedPGPCLIP(prompt_args, prompt_args["class_names"], clip_model)
    elif args.method == "FedOTP":
        from src.models.clip_fedotp import FedOTPCLIP

        args.num_prompt = 2
        prompt_args = {
            "ctx_init": args.ctx_init,
            "class_specific_context": args.csc,
            "num_ctx": args.num_ctx,
            "class_token_position": args.class_token_position,
            "class_names": args.class_names,
            "num_prompt": args.num_prompt,
            "OT": args.OT,
            "top_percent": args.top_percent,
            "eps": args.eps,
            "thresh": args.thresh,
            "max_iter": args.max_iter,
        }
        backbone = FedOTPCLIP(prompt_args, prompt_args["class_names"], clip_model)

    elif args.method == "FedLoCoOp":
        from src.models.clip_locoop import LoCoOpCLIP

        prompt_args = {
            "num_ctx": args.num_ctx,
            "class_specific_context": args.csc,
            "ctx_init": args.ctx_init,
            "class_names": args.class_names,
            "ctx_position": args.class_token_position,
        }
        backbone = LoCoOpCLIP(prompt_args, prompt_args["class_names"], clip_model)

    elif args.method == "FedGalLop":
        from src.models.clip_gallop import GalLoPCLIP

        args.ctx_init = "A photo of a {}"
        prompt_args = {
            "num_ctx": args.num_ctx,
            "ctx_init": args.ctx_init,
            "top_k": [5, 10, 15, 20],
            "use_local_features": True,
            "learn_local_prompts": True,
            "learn_global_prompts": True,
            "learn_local_proj": True,
            "n_global_prompts": args.num_prompt,
            "n_local_prompts": args.num_local_prompt,
            "prompts_batch_size": args.prompts_batch_size,
            "class_names": args.class_names,
        }
        backbone = GalLoPCLIP(prompt_args, prompt_args["class_names"], clip_model)
        backbone.initialize_prompt()
        backbone.freeze_clip()

    elif args.method == "FedLAPT":
        from src.models.clip_fedlapt import CoOp_NegOODPrompt

        prompt_args = {
            "num_ctx": args.num_ctx,
            "ctx_init": None,
            "ctx_position": args.class_token_position,
            "prompttype": args.prompttype,
            "num_ex_prompt": args.num_ex_prompt,
            "text_prompt": "tip",
            "preprocess": preprocess,
            "class_names": args.class_names,
            "image_size": 224,
            "dataset": args.id_dataset,
            "class_specific_context": args.csc,
            "text_center": args.text_center,
        }
        backbone = CoOp_NegOODPrompt(prompt_args, prompt_args["class_names"], clip_model)

    elif args.method == "FOCoOp":
        from src.models.clip_focoop import FOCoOpCLIP

        args.csc = True
        prompt_args = {
            "num_ctx": args.num_ctx,
            "ctx_init": args.ctx_init,
            "ctx_position": args.class_token_position,
            "class_specific_context": args.csc,
            "num_ex_prompt": args.num_ex_prompt,
            "frac": args.frac,
            "device": args.device,
            "prompttype": args.prompttype,
            "dataset": args.id_dataset,
            "class_names": args.class_names,
            "num_prompt": args.num_prompt,
        }
        backbone = FOCoOpCLIP(prompt_args, prompt_args["class_names"], clip_model)
    else:
        raise NotImplementedError(f'The method "{args.method}" is not implemented')

    print("Turning off gradients in both the image and the text encoder")
    for name, param in backbone.named_parameters():
        if "prompt_learner" in name:
            param.requires_grad_(True)
        elif name in ["global_prompt", "local_prompts", "local_proj.linear.weight"]:  # customized for GalLop
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return backbone


def make_save_path(args):
    root_path = os.path.dirname(__file__)
    root_path = os.path.join(os.path.dirname(root_path), "results")
    if args.id_dataset in ["PACS", "office", "domainnet"]:
        dataset_setting = f"{args.id_dataset}_{args.leave_out}"
    else:
        if args.pathological:
            if args.non_overlap:
                dataset_setting = f"{args.id_dataset}_pathological{args.class_per_client}_non_overlap_{args.num_client}clients_{args.num_shot}shots"
            else:
                dataset_setting = f"{args.id_dataset}_pathological{args.class_per_client}_{args.num_client}clients_{args.num_shot}shots"
        else:
            dataset_setting = f"{args.id_dataset}_{args.alpha}alpha_{args.num_client}clients_{args.num_shot}shots"
    task_path = dataset_setting + f"/{args.method}_customclip"
    save_path = os.path.abspath(os.path.join(root_path, task_path))
    os.makedirs(save_path, exist_ok=True)
    return save_path


def get_server_and_client(args, client_args, server_args):
    if args.method == "PromptFL":
        from src.algorithms.PromptFL.client_promptfl import PromptFLClient
        from src.algorithms.PromptFL.server_promptfl import PromptFLServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
        server = PromptFLServer
        client = PromptFLClient

    elif args.method == "PromptFolio":
        from src.algorithms.PromptFolio.client_promptfolio import PromptFolioClient
        from src.algorithms.PromptFolio.server_promptfolio import PromptFolioServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay

        server = PromptFolioServer
        client = PromptFolioClient

    elif args.method == "FedPGP":
        from src.algorithms.FedPGP.client_fedpgp import FedPGPClient
        from src.algorithms.FedPGP.server_fedpgp import FedPGPServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
            client_args[cid]["temp"] = args.temp
            client_args[cid]["mu"] = args.mu
        server = FedPGPServer
        client = FedPGPClient

    elif args.method == "FedOTP":
        from src.algorithms.FedOTP.client_fedotp import FedOTPClient
        from src.algorithms.FedOTP.server_fedotp import FedOTPServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
        server = FedOTPServer
        client = FedOTPClient

    elif args.method == "FedLoCoOp":
        from src.algorithms.FedLoCoOp.client_fedlocoop import FedLoCoOpClient
        from src.algorithms.FedLoCoOp.server_fedlocoop import FedLoCoOpServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
            client_args[cid]["n_cls"] = len(args.class_names)
            client_args[cid]["precision"] = args.precision
            client_args[cid]["top_k"] = args.top_k
            client_args[cid]["lambda_local"] = args.lambda_local
        client = FedLoCoOpClient
        server = FedLoCoOpServer

    elif args.method == "FedGalLop":
        from src.algorithms.FedGalLop.client_fedgallop import FedGalLopClient
        from src.algorithms.FedGalLop.server_fedgallop import FedGalLopServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
            client_args[cid]["n_cls"] = len(args.class_names)
            client_args[cid]["precision"] = args.precision
            client_args[cid]["top_k"] = args.top_k
            client_args[cid]["lambda_local"] = args.lambda_local
            client_args[cid]["use_global_loss"] = True
            client_args[cid]["use_local_loss"] = True
            client_args[cid]["global_dropout_p"] = 0.9

        client = FedGalLopClient
        server = FedGalLopServer

    elif args.method == "FedLAPT":
        from src.algorithms.FedLAPT.client_fedlapt import FedLAPTClient
        from src.algorithms.FedLAPT.server_fedlapt import FedLAPTServer

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
            client_args[cid]["n_cls"] = len(args.class_names)
            client_args[cid]["alpha"] = args.alpha_mixup
            client_args[cid]["beta"] = args.beta_mixup
            client_args[cid]["total_gs_num"] = args.total_gs_num
            client_args[cid]["selected_gs_num"] = args.selected_gs_num
            client_args[cid]["gs_loss_weight"] = args.gs_loss_weight
            client_args[cid]["loss_weights"] = args.loss_weights
            client_args[cid]["soft_split"] = args.soft_split
            client_args[cid]["mix_strategy"] = args.mix_strategy
            client_args[cid]["loss_components"] = args.loss_components
            client_args[cid]["use_gs"] = args.use_gs
            client_args[cid]["num_ex_prompt"] = args.num_ex_prompt
            client_args[cid]["queue_capacity"] = args.queue_capacity
            client_args[cid]["pre_queue"] = args.pre_queue
            client_args[cid]["iter_recomputation"] = args.iter_recomputation
        client = FedLAPTClient
        server = FedLAPTServer

    elif args.method == "FOCoOp":
        from src.algorithms.FOCoOp.client_focoop import FOCoOpClient
        from src.algorithms.FOCoOp.server_focoop import FOCoOpServer

        server_args["ema"] = args.ema
        server_args["num_ex_prompt"] = args.num_ex_prompt
        server_args["uot"] = args.uot

        for cid in range(len(client_args)):
            client_args[cid]["learning_rate"] = args.learning_rate
            client_args[cid]["momentum"] = args.momentum
            client_args[cid]["weight_decay"] = args.weight_decay
            client_args[cid]["n_cls"] = len(args.class_names)
            client_args[cid]["num_ex_prompt"] = args.num_ex_prompt
            client_args[cid]["num_prompt"] = args.num_prompt
            client_args[cid]["dro"] = args.dro
            client_args[cid]["gamma_1"] = args.gamma_1
            client_args[cid]["gamma_2"] = args.gamma_2
            client_args[cid]["tau"] = args.tau
            client_args[cid]["iter_id"] = args.iter_id
            client_args[cid]["iter_ood"] = args.iter_ood
            client_args[cid]["noise_strength"] = args.noise_strength
        client = FOCoOpClient
        server = FOCoOpServer
    else:
        raise NotImplementedError("method not support")

    return server, client, client_args, server_args


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
