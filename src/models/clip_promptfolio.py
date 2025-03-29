import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from src.utils.clip_utils import TextEncoder

_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    def __init__(self, prompt_args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = prompt_args["ctx_init"]
        ctx_suf_init = None
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.class_specific_context = prompt_args["class_specific_context"]
        self.num_prompt = prompt_args["num_prompt"]
        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        if ctx_init is not None:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt_prefix = ctx_init
        else:
            n_ctx = prompt_args["num_ctx"]
            prompt_prefix = " ".join(["X"] * n_ctx)

        if ctx_suf_init:
            prompt_suffix = " " + ctx_suf_init
        else:
            prompt_suffix = ""

        prompts = [prompt_prefix + " " + name + prompt_suffix + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.num_prompt, 1)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        if self.class_specific_context:
            if ctx_init:
                ctx_vectors = embedding[:, 1 : 1 + n_ctx, :]
            else:
                ctx_vectors = torch.empty(n_cls * self.num_prompt, n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
            ctx = nn.Parameter(ctx_vectors)
        else:
            if ctx_init:
                ctx_vectors = embedding[: self.num_prompt, 1 : 1 + n_ctx, :]
            else:
                # ctx_vectors = torch.empty(self.num_prompt, n_ctx, ctx_dim, dtype=dtype)
                # nn.init.normal_(ctx_vectors, std=0.02)
                ctx_vectors_global = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors_local = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors_global, std=0.02)
                nn.init.normal_(ctx_vectors_local, std=0.02)
            # ctx = nn.Parameter(ctx_vectors)
            ctx_global = nn.Parameter(ctx_vectors_global)
            ctx_local = nn.Parameter(ctx_vectors_local)

        self.ctx_global = ctx_global
        self.ctx_local = ctx_local

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = prompt_args["class_token_position"]

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

    def forward(self):
        ctx = torch.stack((self.ctx_global, self.ctx_local), dim=0)
        if not self.class_specific_context:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            ctx = ctx.permute(1, 0, 2, 3)
            ctx = ctx.contiguous().view(self.num_prompt * self.n_cls, self.n_ctx, ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )

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


class PromptFolioCLIP(nn.Module):
    def __init__(self, prompt_args, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(prompt_args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.num_prompt = prompt_args["num_prompt"]
        self.frac = prompt_args["frac"]
        self.n_cls = len(classnames)

    def forward(self, image):

        image_features = self.get_img_features(image)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features0 = self.text_encoder(prompts[: self.n_cls], tokenized_prompts[: self.n_cls])
        text_features1 = self.text_encoder(
            prompts[self.n_cls : 2 * self.n_cls], tokenized_prompts[self.n_cls : 2 * self.n_cls]
        )

        text_features0 = text_features0 / text_features0.norm(dim=-1, keepdim=True)
        text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        # frac = 0 means fully global, frac = 1 means fully local
        text_features = (1 - self.frac) * text_features0 + self.frac * text_features1
        logits = logit_scale * image_features @ text_features.t()

        return logits

    def get_img_features(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        if image_features.shape[0] != image.shape[0]:
            image_features = image_features.permute(1, 0, 2)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features
