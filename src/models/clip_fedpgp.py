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
        n_ctx = prompt_args["num_ctx"]
        ctx_init = prompt_args["ctx_init"]
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.class_specific_context = prompt_args["class_specific_context"]

        bottleneck = prompt_args["bottleneck"]
        self.num_prompt = prompt_args["num_prompt"]

        if ctx_init is not None:

            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if self.class_specific_context:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                U = torch.empty(self.num_prompt, n_ctx, bottleneck, dtype=dtype)
                V = torch.empty(self.num_prompt, bottleneck, ctx_dim, dtype=dtype)
                sigma = torch.empty(self.num_prompt, n_ctx, ctx_dim, dtype=dtype)
                # ctx_vectors = torch.matmul(U,V)

            nn.init.normal_(U, std=0.02)
            nn.init.normal_(V, std=0.02)
            nn.init.normal_(sigma, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)
        self.sigma = nn.Parameter(sigma)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.num_prompt, 1)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])
        self.register_buffer("embedding", embedding)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = prompt_args["class_token_position"]

    def forward(self):

        U = self.U
        V = self.V
        UV = torch.matmul(U, V)
        sigma = self.sigma
        ctx = UV + self.sigma
        embedding = self.embedding

        if not self.class_specific_context:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            ctx = ctx.permute(1, 0, 2, 3)
            ctx = ctx.contiguous().view(self.num_prompt * self.n_cls, self.n_ctx, ctx.shape[3])

            UV = UV.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            UV = UV.permute(1, 0, 2, 3)
            UV = UV.contiguous().view(self.num_prompt * self.n_cls, self.n_ctx, UV.shape[3])

            sigma = sigma.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            sigma = sigma.permute(1, 0, 2, 3)
            sigma = sigma.contiguous().view(self.num_prompt * self.n_cls, self.n_ctx, sigma.shape[3])

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
            prompts_sigma = torch.cat(
                [
                    prefix,
                    sigma,
                    suffix,
                ],
                dim=1,
            )
            prompts_UV = torch.cat(
                [
                    prefix,
                    UV,
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

        return embedding, prompts_sigma, prompts_UV, prompts


class FedPGPCLIP(nn.Module):
    def __init__(self, prompt_args, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(prompt_args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        embedding, prompts_sigma, prompts_UV, prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if self.training == True:
            text_features_0 = self.text_encoder(embedding, tokenized_prompts)
            text_features_sigma = self.text_encoder(prompts_sigma, tokenized_prompts)
            text_features_UV = self.text_encoder(prompts_UV, tokenized_prompts)

            text_features_0 = text_features_0 / text_features_0.norm(dim=-1, keepdim=True)
            text_features_sigma = text_features_sigma / text_features_sigma.norm(dim=-1, keepdim=True)
            text_features_UV = text_features_UV / text_features_UV.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

            return text_features_0, text_features_sigma, text_features_UV, text_features, logits

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
