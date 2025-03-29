import json
import os

import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm

from src.models.clip_for_wordnet_prepare import generate_cossim_idname_wordnet_dedup
from src.prompt.prompt import get_templates
from src.utils.clip_utils import TextEncoder

_tokenizer = _Tokenizer()


def get_selected_ood_text_list(dataset, classnames, total_ood_num=100, clip_model=None):
    print("the ID dataset is:", dataset)
    print("total_ood_num is:", total_ood_num)
    foot_path = "../data/txtfiles_output/"
    if not os.path.exists(foot_path):
        os.makedirs(foot_path)
    wordnet_processed_path = foot_path + "wordnet_" + dataset + "_cossim_dedup.pth"
    if os.path.exists(wordnet_processed_path):
        wordnet_dict = torch.load(wordnet_processed_path)
    else:
        generate_cossim_idname_wordnet_dedup(classnames, wordnet_processed_path, clip_model=clip_model)
        wordnet_dict = torch.load(wordnet_processed_path)

    can_list_adj = wordnet_dict["text_list_adj"]
    can_cos_adj = wordnet_dict["cos_sim_adj"]
    can_list_noun = wordnet_dict["text_list_noun"]
    can_cos_noun = wordnet_dict["cos_sim_noun"]

    adj_num = int(total_ood_num * (len(can_list_adj) / (len(can_list_adj) + len(can_list_noun))))
    noun_num = total_ood_num - adj_num

    cate_num = can_cos_adj.size(1)
    cos_sim_indice_selected = int(cate_num * 0.95)

    value_cos_adj, _ = can_cos_adj.sort(1)
    value_cos_noun, _ = can_cos_noun.sort(1)

    selected_value_cos_adj = value_cos_adj[:, cos_sim_indice_selected]
    selected_value_cos_noun = value_cos_noun[:, cos_sim_indice_selected]
    value_sim_adj, indice_sim_adj = selected_value_cos_adj.sort(0)
    value_sim_noun, indice_sim_noun = selected_value_cos_noun.sort(0)

    selected_adj_text = [can_list_adj[i] for i in indice_sim_adj[:adj_num]]
    selected_noun_text = [can_list_noun[i] for i in indice_sim_noun[:noun_num]]

    unselected_adj_text = [can_list_adj[i] for i in indice_sim_adj[adj_num:]]
    unselected_noun_text = [can_list_noun[i] for i in indice_sim_noun[noun_num:]]

    return selected_adj_text, selected_noun_text, unselected_adj_text, unselected_noun_text


class PromptLearner(nn.Module):
    def __init__(self, prompt_args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = prompt_args["num_ctx"]
        self.n_ex_prompts = prompt_args["num_ex_prompt"]
        ctx_init = prompt_args["ctx_init"]
        prompttype = prompt_args["prompttype"]
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = prompt_args["image_size"]  # 224

        class_specific_context = prompt_args["class_specific_context"]  # False
        ctx_position = prompt_args["ctx_position"]

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init is not None:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding_temp = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding_temp[0, 1 : 1 + n_ctx, :]
            ctx_ood_vectors = embedding_temp[0, 1 : 1 + n_ctx, :].clone()
            prompt_prefix = ctx_init
            ood_prompt_prefix = ctx_init
            self.ctx = nn.Parameter(ctx_vectors)
            self.ctx_ood = nn.Parameter(ctx_ood_vectors)

            if class_specific_context:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ctx_ood_vectors = torch.empty(self.n_ex_prompts, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                nn.init.normal_(ctx_ood_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
                ood_prompt_prefix = " ".join(["X"] * (n_ctx + 1))
                self.ctx = nn.Parameter(ctx_vectors)
                self.ctx_ood = nn.Parameter(ctx_ood_vectors)
            else:
                if prompttype == "dis_aware":
                    print("Initializing a distribution aware context")
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                    ctx_ood_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                    nn.init.normal_(ctx_vectors, std=0.02)
                    nn.init.normal_(ctx_ood_vectors, std=0.02)
                    prompt_prefix = " ".join(["X"] * n_ctx)
                    ood_prompt_prefix = " ".join(["X"] * (n_ctx + 1))
                    self.ctx = nn.Parameter(ctx_vectors)
                    self.ctx_ood = nn.Parameter(ctx_ood_vectors)
                elif prompttype == "unified":
                    ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                    nn.init.normal_(ctx_vectors, std=0.02)
                    ctx_ood_vectors = ctx_vectors
                    prompt_prefix = " ".join(["X"] * n_ctx)
                    ood_prompt_prefix = " ".join(["X"] * (n_ctx + 1))
                    self.ctx = nn.Parameter(ctx_vectors)
                    self.ctx_ood = self.ctx
                elif prompttype == "class_specific":
                    ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                    ctx_ood_vectors = torch.empty(self.n_ex_prompts, n_ctx, ctx_dim, dtype=dtype)
                    nn.init.normal_(ctx_vectors, std=0.02)
                    nn.init.normal_(ctx_ood_vectors, std=0.02)
                    prompt_prefix = " ".join(["X"] * n_ctx)
                    ood_prompt_prefix = " ".join(["X"] * (n_ctx + 1))
                    self.ctx = nn.Parameter(ctx_vectors)
                    self.ctx_ood = nn.Parameter(ctx_ood_vectors)
                else:
                    raise NotImplementedError

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        selected_adj_text, selected_noun_text, unselected_adj_text, unselected_noun_text = get_selected_ood_text_list(
            prompt_args["dataset"], classnames, total_ood_num=self.n_ex_prompts, clip_model=clip_model
        )

        selected_ood_text = selected_adj_text + selected_noun_text
        ood_prompts = [prompt_prefix + " " + name + "." for name in selected_ood_text]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        ood_tokenized_prompts = torch.cat([clip.tokenize(p) for p in ood_prompts])
        with torch.no_grad():
            ood_embedding = clip_model.token_embedding(ood_tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.register_buffer("ood_token_prefix", ood_embedding[:, :1, :])
        self.register_buffer("ood_token_suffix", ood_embedding[:, 1 + n_ctx :, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        self.tokenized_prompts = torch.cat((tokenized_prompts, ood_tokenized_prompts), dim=0)
        self.name_lens = name_lens
        self.class_token_position = ctx_position  # end

    def forward(self):
        ctx_vanilla = self.ctx
        ctx_ood_vanilla = self.ctx_ood
        if ctx_vanilla.dim() == 2:
            ctx = ctx_vanilla.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_ood = ctx_ood_vanilla.unsqueeze(0).expand(self.n_ex_prompts, -1, -1)
        else:
            ctx = ctx_vanilla
            ctx_ood = ctx_ood_vanilla

        prefix = self.token_prefix
        suffix = self.token_suffix

        ood_prefix = self.ood_token_prefix
        ood_suffix = self.ood_token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )
            # pdb.set_trace()
            ood_prompts = torch.cat(
                [
                    ood_prefix,
                    ctx_ood,
                    ood_suffix,
                ],
                dim=1,
            )
            prompts = torch.cat((prompts, ood_prompts), dim=0)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, prompt_args, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(prompt_args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.vanilla_clip = clip_model.cuda()
        self.text_features = None

    def forward(self, image, return_feat):
        image_features = self.image_encoder(image.type(self.dtype))  ##128*512

        prompts = self.prompt_learner()  # torch.Size([1000, 77, 512])
        tokenized_prompts = self.tokenized_prompts  ## 1000*77
        text_features = self.text_encoder(prompts, tokenized_prompts)  # 1000*512
        self.text_features = text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        if return_feat:
            return image_features, text_features, logit_scale
        else:
            logits = logit_scale * image_features @ text_features.t()
            return logits


def get_text_features_neg(clip_model, dataset, classnames, text_prompt, text_center, ood_num):
    templates = get_templates(text_prompt)

    print("adopt text prompt of", text_prompt)
    with torch.no_grad():
        text_features = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]
            if "cupl" in text_prompt:
                cupl_file = "CuPL_prompts_imagenet.json"
                f = open("./openood/networks/clip/gpt3_prompts/" + cupl_file)
                cupl_prompts = json.load(f)
                texts += cupl_prompts[classname]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if text_center:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_features.append(class_embedding)
            else:
                text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=1).cuda()

    selected_adj_text, selected_noun_text, unselected_adj_text, unselected_noun_text = get_selected_ood_text_list(
        dataset, classnames, total_ood_num=ood_num
    )
    with torch.no_grad():
        text_features_neg = []
        for classname in tqdm(selected_adj_text):
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if text_center:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_features_neg.append(class_embedding)
            else:
                text_features_neg.append(class_embeddings)

        for classname in tqdm(selected_noun_text):
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            if text_center:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                text_features_neg.append(class_embedding)
            else:
                text_features_neg.append(class_embeddings)

        text_features_unselected = text_features_neg
        text_features_unselected = torch.stack(text_features_unselected, dim=1).cuda()
        text_features_neg = torch.stack(text_features_neg, dim=1).cuda()

    text_features = torch.cat((text_features, text_features_neg), dim=1)
    return text_features.transpose(0, 1), text_features_unselected.transpose(0, 1)


class CoOp_NegOODPrompt(CustomCLIP):
    def __init__(self, prompt_args, classnames, clip_model):
        super().__init__(prompt_args, classnames, clip_model)
        self.n_ex_prompts = prompt_args["num_ex_prompt"]
        self.text_prompt = prompt_args["text_prompt"]
        self.n_cls = len(classnames)

        self.n_output = self.n_cls + int(self.n_ex_prompts)

        print("Building custom CLIP")
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.text_center = prompt_args["text_center"]
        self.text_features, self.text_features_unselected = get_text_features_neg(
            clip_model, prompt_args["dataset"], classnames, self.text_prompt, self.text_center, self.n_ex_prompts
        )
        print("shape of pre-computed text features:", self.text_features.shape)

    def forward(self, x, return_feat=False):
        return super().forward(x, return_feat)

        # image_features = self.model.encode_image(x)
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ self.zeroshot_weights
        # return logits
