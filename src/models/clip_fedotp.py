import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.nn import functional as F

from src.utils.clip_utils import TextEncoder

_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    def __init__(self, prompt_args, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = prompt_args["ctx_init"]

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.class_specific_context = prompt_args["class_specific_context"]
        self.N = prompt_args["num_prompt"]
        n_ctx = prompt_args["num_ctx"]

        classnames = [name.replace("_", " ") for name in classnames]

        self.name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            if self.class_specific_context:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                self.N = 1
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.class_token_position = prompt_args["class_token_position"]
        self.name_lens = name_lens

    def forward(self):

        ctx = self.ctx

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[3])

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


class FedOTPCLIP(nn.Module):
    def __init__(self, prompt_args, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = PromptLearner(prompt_args, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.N = prompt_args["num_prompt"]

        self.OT = prompt_args["OT"]
        self.top_percent = prompt_args["top_percent"]
        self.eps = prompt_args["eps"]
        self.thresh = prompt_args["thresh"]
        self.max_iter = prompt_args["max_iter"]

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = self.thresh
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T

    def entropic_COT_fast(self, a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False):
        """
        modify from ot.partial.entropic_partial_wasserstein in torch version

        """
        dx = torch.ones_like(a)
        dy = torch.ones_like(b)

        log_e = {"err": []}
        stopThr = self.thresh

        K = M

        Kp = torch.matmul(torch.diag_embed(1 / a, dim1=1), K)
        Kq = torch.matmul(torch.diag_embed(1 / b, dim1=1), K.permute(0, 2, 1))

        err, cpt = 1, 0
        u = dx
        v = dy
        while cpt < numItermax:

            v0 = v
            temp = torch.div(dx, torch.matmul(Kp, v.unsqueeze(-1)).squeeze(-1))
            u = torch.minimum(temp, dx)
            v = torch.div(dy, torch.matmul(Kq, u.unsqueeze(-1)).squeeze(-1))

            cpt = cpt + 1
            err = (v - v0).abs().mean()
            if err.item() < stopThr:
                break
        Kprev = torch.matmul(torch.diag_embed(u, dim1=1), K)
        Kprev = torch.matmul(Kprev, torch.diag_embed(v, dim1=1))
        if log:
            return Kprev, log_e
        else:
            return Kprev

    def forward(self, image):

        b = image.shape[0]
        image_features = self.image_encoder(image.type(self.dtype))
        image_feature_pool = image_features[0]
        image_features = image_features[1:]
        M = image_features.shape[0]
        self.d = image_features.shape[-1]

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.contiguous().view(self.N, self.n_cls, self.d)

        image_features = F.normalize(image_features, dim=2)
        text_features = F.normalize(text_features, dim=2)

        sim = torch.einsum("mbd,ncd->mnbc", image_features, text_features).contiguous()
        sim = sim.view(M, self.N, b * self.n_cls)
        sim = sim.permute(2, 0, 1)
        wdist = 1.0 - sim

        xx = torch.zeros(b * self.n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1.0 / M)
        if self.OT == "Sinkhorn":
            yy = torch.zeros(b * self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1.0 / self.N)
        elif self.OT == "COT":
            top_percent = min(torch.sum(xx).item(), self.top_percent)
            yy = (
                torch.zeros(b * self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1.0 / self.N)
                * top_percent
            )

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            if self.OT == "Sinkhorn":
                T = self.Sinkhorn(KK, xx, yy)
            elif self.OT == "COT":
                T = self.entropic_COT_fast(xx, yy, KK, 0.01, numItermax=self.max_iter)
        if torch.isnan(T).any():
            return None

        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b, self.n_cls)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * sim_op

        return logits
