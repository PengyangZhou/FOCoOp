import clip
import torch
import torch.nn as nn
from clip import tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from src.models.clip_fedlapt import get_selected_ood_text_list
from src.utils.clip_utils import TextEncoder

_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    def __init__(self, prompt_args, classnames, clip_model):

        super().__init__()
        num_ctx = prompt_args["num_ctx"]
        ctx_init = prompt_args["ctx_init"]
        self.ctx_position = prompt_args["ctx_position"]
        self.class_specific_context = prompt_args["class_specific_context"]
        self.dataset = prompt_args["dataset"]
        self.frac = prompt_args["frac"]
        self.num_prompt = prompt_args["num_prompt"]
        n_cls = len(classnames)
        n_ood_cls = prompt_args["num_ex_prompt"]
        self.num_ex_prompt = prompt_args["num_ex_prompt"]
        dtype = clip_model.dtype
        self.dtype = dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.ctx_dim = ctx_dim
        prompttype = prompt_args["prompttype"]

        if ctx_init is not None:
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ").replace("{}", " ")

            if "[CLS]" in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                self.ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            num_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_local_vectors = embedding[0, 1 : 1 + num_ctx, :]
            ctx_global_vectors = embedding[0, 1 : 1 + num_ctx, :].clone()
            ctx_ood_vectors = embedding[0, 1 : 1 + num_ctx, :].clone()
            prompt_prefix = ctx_init
            ood_prompt_prefix = ctx_init
            self.ctx_local = nn.Parameter(ctx_local_vectors)
            self.ctx_global = nn.Parameter(ctx_global_vectors)
            self.ctx_ood = nn.Parameter(ctx_ood_vectors)

        else:

            prompt_prefix = " ".join(["X"] * num_ctx)
            ood_prompt_prefix = " ".join(["X"] * num_ctx)

            if self.class_specific_context:
                print("Initializing class-specific context vectors.")
                ctx_local_vectors = torch.empty(n_cls, num_ctx, ctx_dim, dtype=dtype)
                ctx_ood_vectors = torch.empty(n_ood_cls, num_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_local_vectors, std=0.02)
                nn.init.normal_(ctx_ood_vectors, std=0.02)
                ctx_global_vectors = ctx_local_vectors.clone()

                self.ctx_local = nn.Parameter(ctx_local_vectors)
                self.ctx_global = nn.Parameter(ctx_global_vectors)
                self.ctx_ood = nn.Parameter(ctx_ood_vectors)

            else:
                if prompttype == "dis_aware":
                    print("Initializing distribution aware context vectors.")
                    ctx_local_vectors = torch.empty(num_ctx, ctx_dim, dtype=dtype)
                    ctx_ood_vectors = torch.empty(num_ctx, ctx_dim, dtype=dtype)
                    nn.init.normal_(ctx_local_vectors, std=0.02)
                    nn.init.normal_(ctx_ood_vectors, std=0.02)
                    ctx_global_vectors = ctx_local_vectors.clone()

                    self.ctx_local = nn.Parameter(ctx_local_vectors)
                    self.ctx_global = nn.Parameter(ctx_global_vectors)
                    self.ctx_ood = nn.Parameter(ctx_ood_vectors)

                elif prompttype == "unified":
                    print("Initializing unified context vectors.")
                    ctx_local_vectors = torch.empty(num_ctx, ctx_dim, dtype=dtype)
                    nn.init.normal_(ctx_local_vectors, std=0.02)

                    self.ctx_local = nn.Parameter(ctx_local_vectors)
                    self.ctx_global = self.ctx_local
                    self.ctx_ood = self.ctx_local
                else:
                    raise NotImplementedError

        print(f'Initial context: "{prompt_prefix}" ')
        print(f"Number of context words (tokens): {num_ctx}")

        self.ctx_init_state = ctx_local_vectors.detach().clone()
        self.ctx_init_state._requires_grad = False

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        selected_adj_text, selected_noun_text, unselected_adj_text, unselected_noun_text = get_selected_ood_text_list(
            self.dataset, classnames, total_ood_num=self.num_ex_prompt, clip_model=clip_model
        )

        selected_ood_text = selected_adj_text + selected_noun_text
        ood_prompts = [ood_prompt_prefix + " " + name + "." for name in selected_ood_text]
        ood_name_lens = [len(_tokenizer.encode(name)) for name in selected_ood_text]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_ood_prompts = torch.cat([clip.tokenize(p) for p in ood_prompts])

        self.tokenized_prompts = torch.cat([tokenized_prompts, tokenized_ood_prompts], dim=0)
        ID_size = len(prompts)
        with torch.no_grad():
            embedding = clip_model.token_embedding(self.tokenized_prompts).type(dtype)

        self.register_buffer("id_token_prefix", embedding[:ID_size, :1, :])
        self.register_buffer("id_token_suffix", embedding[:ID_size, 1 + num_ctx :, :])
        self.register_buffer("ood_token_prefix", embedding[-self.num_ex_prompt :, :1, :])
        self.register_buffer("ood_token_suffix", embedding[-self.num_ex_prompt :, 1 + num_ctx :, :])

        self.n_cls = n_cls
        self.num_ctx = num_ctx
        self.name_lens = name_lens
        self.ood_name_lens = ood_name_lens

        self.ctx_init = ctx_init
        self.classnames = classnames
        self.prompt_prefix = prompt_prefix
        self.ood_prompt_prefix = ood_prompt_prefix
        self.ctx_init_state = ctx_local_vectors.detach().clone()

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors)

    def reset_classnames(self, classnames, clip_model):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()

        selected_adj_text, selected_noun_text, unselected_adj_text, unselected_noun_text = get_selected_ood_text_list(
            self.datasets, classnames, total_ood_num=self.num_ex_prompt, clip_model=clip_model
        )

        selected_ood_text = selected_adj_text + selected_noun_text
        ood_prompts = [self.ood_prompt_prefix + " " + name + "." for name in selected_ood_text]
        prompts = torch.cat((prompts, ood_prompts), dim=0)
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.id_token_prefix = embedding[: self.n_cls, :1, :]
        self.id_token_suffix = embedding[: self.n_cls, 1 + self.num_ctx :, :]
        self.ood_token_prefix = embedding[self.n_cls :, :1, :]
        self.ood_token_suffix = embedding[self.n_cls :, 1 + self.num_ctx :, :]
        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self):
        ctx_local = self.ctx_local
        ctx_global = self.ctx_global
        ctx_ood = self.ctx_ood

        if ctx_local.dim() == 2:
            ctx_local = ctx_local.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_global = ctx_global.unsqueeze(0).expand(self.n_cls, -1, -1)
        if ctx_ood.dim() == 2:
            ctx_ood = ctx_ood.unsqueeze(0).expand(self.num_ex_prompt, -1, -1)

        ctx = (1 - self.frac) * ctx_local + self.frac * ctx_global

        id_prefix = self.id_token_prefix
        id_suffix = self.id_token_suffix

        ood_prefix = self.ood_token_prefix
        ood_suffix = self.ood_token_suffix

        if self.ctx_position:
            assert self.ctx_position == "end"

        if self.ctx_position == "end":
            id_local_prompt = torch.cat(
                [
                    id_prefix,
                    ctx,
                    id_suffix,
                ],
                dim=-2,
            )

            ood_prompt = torch.cat(
                [
                    ood_prefix,
                    ctx_ood,
                    ood_suffix,
                ],
                dim=-2,
            )
            prompts = torch.cat((id_local_prompt, ood_prompt), dim=0)
        elif self.ctx_position == "middle":
            if self.split_idx is not None:
                half_num_ctx = self.split_idx
            else:
                half_num_ctx = self.num_ctx // 2
            id_local_prompts = []
            id_global_prompts = []
            ood_prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = id_prefix[i : i + 1, :, :]
                class_i = id_suffix[i : i + 1, :name_len, :]
                suffix_i = id_suffix[i : i + 1, name_len:, :]
                ctx_i_local_half1 = ctx[i : i + 1, :half_num_ctx, :]
                ctx_i_local_half2 = ctx[i : i + 1, half_num_ctx:, :]

                id_local_prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_local_half1,
                        class_i,
                        ctx_i_local_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                id_local_prompts.append(id_local_prompt)

            for i in range(self.n_ood_cls):
                name_len = self.ood_name_lens[i]
                prefix_i = ood_prefix[i : i + 1, :, :]
                class_i = ood_suffix[i : i + 1, :name_len, :]
                suffix_i = ood_suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx_ood[i : i + 1, :half_num_ctx, :]
                ctx_i_half2 = ctx_ood[i : i + 1, half_num_ctx:, :]
                ood_prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                ood_prompts.append(ood_prompt)

            prompts = torch.cat((id_local_prompts, ood_prompts), dim=0)

        elif self.ctx_position == "front":
            id_local_prompts = []
            id_global_prompts = []
            ood_prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = id_prefix[i : i + 1, :, :]
                class_i = id_suffix[i : i + 1, :name_len, :]
                suffix_i = id_suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]

                id_local_prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                id_local_prompts.append(id_local_prompt)

            for i in range(self.n_ood_cls):
                name_len = self.ood_name_lens[i]
                prefix_i = ood_prefix[i : i + 1, :, :]
                class_i = ood_suffix[i : i + 1, :name_len, :]
                suffix_i = ood_suffix[i : i + 1, name_len:, :]
                ctx_i = ctx_ood[i : i + 1, :, :]
                ood_prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                ood_prompts.append(ood_prompt)

            prompts = torch.cat((id_local_prompts, ood_prompts), dim=0)

        else:
            raise ValueError

        return prompts


class FOCoOpCLIP(nn.Module):
    def __init__(self, prompt_args, classnames, clip_model):
        super(FOCoOpCLIP, self).__init__()
        self.n_cls = len(classnames)
        self.classnames = classnames
        self.num_ex_prompt = prompt_args["num_ex_prompt"]

        self.image_encoder = clip_model.visual
        self.clip_model = clip_model
        self.frac = prompt_args["frac"]
        self.device = prompt_args["device"]

        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale.data
        self.prompt_learner = PromptLearner(prompt_args, classnames, clip_model)
        self.text_features = None
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def get_text_features(self):

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        self.text_features = text_features
        return text_features

    def get_image_features(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        return image_features

    def forward(self, image, return_features=False):
        image_features = self.get_image_features(image)
        text_features = self.get_text_features()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        id_local_features = text_features[: self.n_cls]

        ood_features = text_features[self.n_cls :]

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        if not return_features:

            return logits
        else:
            return logits, id_local_features, ood_features, image_features
