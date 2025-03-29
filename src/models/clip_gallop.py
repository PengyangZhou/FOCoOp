from typing import Dict
from typing import Tuple, Type, List, Optional, Any

import clip
import numpy as np
import torch
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch import Tensor
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from torch.utils.checkpoint import checkpoint

from src.algorithms.FedGalLop.loss import topk_reduce
from src.models.clip_w_local.modified_clip_model_gallop import CLIP
from src.models.clip_w_local.modified_clip_model_gallop import Transformer, VisionTransformer, ModifiedResNet

_tokenizer = _Tokenizer()
NoneType = Type[None]
KwargType = Dict[str, Any]


def get_clip_hyperparams(clip_state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    is_vit = "visual.proj" in clip_state_dict
    if is_vit:
        grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)

        kwargs = {
            "embed_dim": clip_state_dict["text_projection"].shape[1],
            "image_resolution": clip_state_dict["visual.conv1.weight"].shape[-1] * grid_size,
            "vision_layers": len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]
            ),
            "vision_width": clip_state_dict["visual.conv1.weight"].shape[0],
            "vision_patch_size": clip_state_dict["visual.conv1.weight"].shape[-1],
            "context_length": clip_state_dict["positional_embedding"].shape[0],
            "vocab_size": clip_state_dict["token_embedding.weight"].shape[0],
            "transformer_width": clip_state_dict["ln_final.weight"].shape[0],
            "transformer_heads": clip_state_dict["ln_final.weight"].shape[0] // 64,
            "transformer_layers": len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith("transformer.resblocks"))
            ),
        }

    else:

        counts: list = [
            len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]
        ]
        output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        assert output_width**2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]

        kwargs = {
            "embed_dim": clip_state_dict["text_projection"].shape[1],
            "image_resolution": output_width * 32,
            "vision_layers": tuple(counts),
            "vision_width": clip_state_dict["visual.layer1.0.conv1.weight"].shape[0],
            "vision_patch_size": None,
            "context_length": clip_state_dict["positional_embedding"].shape[0],
            "vocab_size": clip_state_dict["token_embedding.weight"].shape[0],
            "transformer_width": clip_state_dict["ln_final.weight"].shape[0],
            "transformer_heads": clip_state_dict["ln_final.weight"].shape[0] // 64,
            "transformer_layers": len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith("transformer.resblocks"))
            ),
        }

    return kwargs


class PromptedTransformer(Transformer):
    def __init__(
        self,
        is_textual: bool = True,
        segments: Optional[int] = 0,
        **kwargs: KwargType,
    ) -> NoneType:
        super().__init__(**kwargs)
        self.is_textual = is_textual
        self.segments = segments

    def replace_context(self, x: Tensor, ctx: Tensor) -> Tensor:
        n_ctx = ctx.shape[0]

        if self.is_textual:
            prefix = x[:1, :, :]
            suffix = x[1 + n_ctx :, :, :]
        else:
            prefix = x[0 : x.shape[0] - n_ctx, :, :]
            suffix = torch.Tensor([]).to(x)

        context = ctx.expand(x.shape[1], -1, -1).permute(1, 0, 2)
        return torch.cat([prefix, context, suffix], dim=0)

    def forward(self, x: Tensor, ctx_vectors: Optional[Tensor] = None, batch_first: bool = False) -> Tensor:
        if batch_first:
            x = x.permute(1, 0, 2)

        if ctx_vectors is None or len(ctx_vectors) == 0:
            if self.segments > 0:
                for i in range(self.layers):
                    if i % (self.layers // self.segments) == 0:
                        x, q, k, v = checkpoint(self.resblocks[i], x)
                    else:
                        x, q, k, v = self.resblocks[i](x)
            else:
                for i in range(self.layers):
                    x, q, k, v = self.resblocks[i](x)
        else:
            for i in range(self.layers):
                x, q, k, v = self.resblocks[i](x)
                if i < len(ctx_vectors):
                    x = self.replace_context(x, ctx_vectors[i])

        if batch_first:
            x = x.permute(1, 0, 2)

        return x, q, k, v


class PromptedVisionTransformer(VisionTransformer):
    def __init__(
        self,
        **kwargs: KwargType,
    ) -> NoneType:
        super().__init__(**kwargs)

        self.transformer = PromptedTransformer(
            width=kwargs["width"], layers=kwargs["layers"], heads=kwargs["heads"], attn_mask=None, is_textual=False
        )

    def forward(self, x: torch.Tensor, ctx_vectors: Optional[nn.ParameterList] = None) -> torch.Tensor:
        if ctx_vectors is None or len(ctx_vectors) == 0:
            return super().forward(x)
        else:
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            zeros = torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([self.class_embedding.to(x.dtype) + zeros, x], dim=1)
            x = x + self.positional_embedding.to(x.dtype)

            context = ctx_vectors[0].expand(x.shape[0], -1, -1)
            x = torch.cat([x, context], dim=1)

            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)
            x, q, k, v = self.transformer(x, ctx_vectors[1:])
            x = x.permute(1, 0, 2)
            v = v.permute(1, 0, 2)

            v = self.ln_post(v)

            v = v[:, 1:]
            B, _, C = x[:, 1:].shape
            v = v.reshape(B, -1, C).contiguous()

            x = self.ln_post(x[:, 0, :])

            if self.proj is not None:
                x = x @ self.proj
                x_local = v @ self.proj
            return x, x_local


class Linear(nn.Module):
    def __init__(self, in_dim: int, identity_init: bool = True) -> NoneType:
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim, bias=False)
        if identity_init:
            nn.init.zeros_(self.linear.weight)
            self.linear.weight.data += torch.eye(in_dim)
        else:
            nn.init.normal_(self.linear.weight, std=in_dim**-0.5)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class GalLoPCLIP(CLIP):
    TRAINABLE_PARAMS: List[str] = []

    def __init__(self, prompt_args, classnames, clip_model):
        clip_state_dict = clip_model.state_dict()

        clip_kwargs = get_clip_hyperparams(clip_state_dict)  #
        clip_kwargs["return_local_features"] = prompt_args["use_local_features"]

        super().__init__(**clip_kwargs)
        self.use_local_features = prompt_args["use_local_features"]
        self.learn_local_proj = prompt_args["learn_local_proj"]

        self.learn_local_prompt = prompt_args["learn_local_prompts"]
        self.learn_global_prompt = prompt_args["learn_global_prompts"]
        self.class_names = classnames  # prompt_args['class_names']
        self.n_global_prompts = prompt_args["n_global_prompts"]
        self.n_local_prompts = prompt_args["n_local_prompts"]
        self.prompts_batch_size = min(prompt_args["prompts_batch_size"], self.n_global_prompts)

        self.topk = prompt_args["top_k"]
        self.template = prompt_args["ctx_init"]

        if isinstance(clip_kwargs["vision_layers"], (tuple, list)):
            clip_model.visual = ModifiedResNet(
                layers=clip_kwargs["vision_layers"],
                output_dim=clip_kwargs["embed_dim"],
                heads=clip_kwargs["vision_width"] * 32 // 64,
                input_resolution=clip_kwargs["image_resolution"],
                width=clip_kwargs["vision_width"],
            )
            vision_dim = clip_kwargs["embed_dim"]
        else:
            clip_model.visual = VisionTransformer(
                input_resolution=clip_kwargs["image_resolution"],
                patch_size=clip_kwargs["vision_patch_size"],
                width=clip_kwargs["vision_width"],
                layers=clip_kwargs["vision_layers"],
                heads=clip_kwargs["vision_width"] // 64,
                output_dim=clip_kwargs["embed_dim"],
            )
            vision_dim = clip_kwargs["vision_width"]

        clip_model.transformer = PromptedTransformer(
            width=clip_kwargs["transformer_width"],
            layers=clip_kwargs["transformer_layers"],
            heads=clip_kwargs["transformer_heads"],
            attn_mask=clip_model.build_attention_mask(),
            segments=8,
        )

        self.local_proj = Linear(vision_dim)

        if self.learn_local_proj:
            self.TRAINABLE_PARAMS.append("local_proj")

        if self.learn_global_prompt or self.learn_local_prompt or self.n_global_prompts > 1 or self.n_local_prompts > 1:
            template = self.template.replace("{}", " ").replace("_", " ").strip()
            tokenized_template = clip.tokenize(template)
            self.template_init_tokens = int(tokenized_template.argmax(dim=-1)) - 1
            self.n_token_context = self.template_init_tokens

            if self.learn_global_prompt or self.n_global_prompts > 1:
                if self.learn_global_prompt:
                    self.TRAINABLE_PARAMS.append("global_prompt")
                self.global_prompt = nn.Parameter(
                    torch.empty(self.n_global_prompts, self.n_token_context, clip_kwargs["transformer_width"]),
                )

            if self.learn_local_prompt or self.n_local_prompts > 1:
                if self.learn_local_prompt:
                    self.TRAINABLE_PARAMS.append("local_prompt")
                self.local_prompts = nn.Parameter(
                    torch.empty(self.n_local_prompts, self.n_token_context, clip_kwargs["transformer_width"]),
                )

        self.initialize_parameters()

        key_issue_clip = self.load_state_dict(clip_state_dict, strict=False)
        if len(key_issue_clip.missing_keys) > 0:
            print(f"Missing keys in CLIP: {key_issue_clip.missing_keys}")

    @torch.no_grad()
    def initialize_prompt(self) -> NoneType:
        if not self.learn_global_prompt and not self.learn_local_prompts:
            return

        template = self.template.replace("{}", " ").replace("_", " ").strip()
        tokenized_template = clip.tokenize(template)
        embedding = self.token_embedding(tokenized_template).type(self.dtype)
        global_prompt_init = embedding[:, 1 : 1 + self.template_init_tokens, :]

        if self.learn_global_prompt:
            self.global_prompt.data[:, : self.template_init_tokens].copy_(
                global_prompt_init.clone().expand(self.n_global_prompts, -1, -1)
            )

        if self.learn_local_prompts:
            self.local_prompts.data[:, : self.template_init_tokens].copy_(
                global_prompt_init.clone().expand(self.n_local_prompts, -1, -1)
            )

    @property
    def num_devices(self) -> int:
        if not hasattr(self, "__device"):
            self.__device = torch.cuda.device_count()
        return self.__device

    def pad_if_necessary(self, x: Tensor) -> Tensor:
        return x, 0

    def unpad_if_necessary(self, x: Tensor, pad: int) -> Tensor:
        if pad == 0:
            return x

        return x[:-pad]

    def _default_encode_text(self, class_names: List[str]) -> Tensor:
        prompts = [self.template.format(name) for name in class_names]
        tokenized_text = clip.tokenize(prompts).to(self.device)
        text_features = super().encode_text(tokenized_text, batch_first=True)
        return text_features.unsqueeze(1)

    def _encode_text(self, prefix: Tensor, prompt: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        x = torch.cat([prefix, prompt, suffix], dim=1)

        x = x + self.positional_embedding.type(self.dtype)

        x, padding = self.pad_if_necessary(x)
        x, *_ = self.transformer(x, batch_first=True)
        x = self.unpad_if_necessary(x, padding)

        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), eot_tokens + self.n_token_context] @ self.text_projection
        return x

    def _single_forward_encode_text(
        self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor
    ) -> Tensor:
        n_prompts = prompts.size(0)
        n_classes = prefix.size(0)

        text_features = self._encode_text(
            prefix.repeat_interleave(n_prompts, dim=0),
            prompts.repeat(n_classes, 1, 1),
            suffix.repeat_interleave(n_prompts, dim=0),
            eot_tokens.repeat_interleave(n_prompts),
        )
        text_features = text_features.unflatten(0, (n_classes, n_prompts))
        return text_features

    def _loop_encode_text(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        text_features = []
        for i in range(prompts.size(0)):
            x = self._encode_text(prefix, prompts[i : i + 1].expand(prefix.size(0), -1, -1), suffix, eot_tokens)
            text_features.append(x)

        return torch.stack(text_features, dim=1)

    def encode_text(self, class_names: List[str]) -> torch.Tensor:
        if not self.learn_global_prompt and not self.learn_local_prompt:
            text_features = self._default_encode_text(class_names)
            return text_features, text_features

        tokenized_text = clip.tokenize(class_names).to(self.device)
        eot_tokens = tokenized_text.argmax(dim=-1)

        with torch.no_grad():
            token_embeddings = self.token_embedding(tokenized_text)

        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 : -self.n_token_context, :]

        if self.learn_global_prompt or self.n_global_prompts > 1:
            global_prompt = self.global_prompt
            if self.prompts_batch_size < self.n_global_prompts and self.training:
                idx_select = torch.randperm(self.n_global_prompts)[: self.prompts_batch_size]
                global_prompt = self.global_prompt[idx_select]
            text_features = self._most_efficient_encode_text(prefix, global_prompt, suffix, eot_tokens)
        else:
            text_features = self._default_encode_text(class_names)

        if self.learn_local_prompt or self.n_local_prompts > 1:
            local_text_features = self._most_efficient_encode_text(prefix, self.local_prompts, suffix, eot_tokens)
        else:
            local_text_features = text_features

        return text_features, local_text_features

    def _most_efficient_encode_text(
        self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor
    ) -> Tensor:
        return self._loop_encode_text(prefix, prompts, suffix, eot_tokens)

    def _default_encode_text(self, class_names: List[str]) -> Tensor:
        prompts = [self.template.format(name) for name in class_names]
        tokenized_text = clip.tokenize(prompts).cuda(non_blocking=True)
        text_features = super().encode_text(tokenized_text, batch_first=True)
        return text_features.unsqueeze(1)

    def encode_image_and_proj(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        image_features, local_features = self.encode_image(image)

        local_features = self.local_proj(local_features)

        if hasattr(self.visual, "proj"):
            image_features = image_features @ self.visual.proj
            if self.use_local_features:
                local_features = local_features @ self.visual.proj

        return image_features, local_features

    def forward(
        self,
        image: Tensor,
        text_features: Optional[Tensor] = None,
        local_text_features: Optional[Tensor] = None,
        return_logits=False,
    ) -> Tensor:
        if self.class_names is not None:
            assert isinstance(self.class_names, list), "class_names must be a list of strings"
        if text_features is not None:
            assert isinstance(text_features, torch.Tensor), "text_features must be a Tensor"
        assert (
            self.class_names is not None or text_features is not None
        ), "Please provide either class_names or text_features"

        if text_features is None:
            assert local_text_features is None, "local_text_features should be None if text_features is None"
            text_features, local_text_features = self.encode_text(self.class_names)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            local_text_features = (
                local_text_features / local_text_features.norm(dim=-1, keepdim=True)
                if self.learn_local_prompt
                else text_features
            )

        image_features, local_features = self.encode_image_and_proj(image)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        global_logits = torch.einsum("bd,kmd-> bkm", image_features, text_features)

        if self.use_local_features:
            local_features = local_features / local_features.norm(dim=-1, keepdim=True)
            local_logits = torch.einsum("bpd,knd-> bpkn", local_features, local_text_features)
        else:
            local_logits = None
        if return_logits:
            return global_logits, local_logits
        else:
            prob_logits, _, _ = self.create_prediction_scores(global_logits, local_logits)
            return prob_logits

    def _prompt_features(self, promtps: Tensor) -> Tensor:
        tokenized_text = clip.tokenize("").to(self.device)
        eot_tokens = tokenized_text.argmax(dim=-1)

        with torch.no_grad():
            token_embeddings = self.token_embedding(tokenized_text)

        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 : -self.n_token_context, :]

        text_features = self._most_efficient_encode_text(prefix, promtps, suffix, eot_tokens)
        return text_features

    def prompt_features(
        self,
    ) -> Tensor:
        global_prompt_features = local_prompt_features = None
        if self.learn_global_prompt:
            global_prompt_features = self._prompt_features(self.global_prompt)

        if self.learn_local_prompt:
            local_prompt_features = self._prompt_features(self.local_prompts)

        return global_prompt_features, local_prompt_features

    @property
    def device(self) -> torch.device:
        return self.text_projection.device

    def freeze_clip(self) -> NoneType:
        for name, p in self.named_parameters():
            if not any([name.startswith(param) for param in self.TRAINABLE_PARAMS]):
                p.requires_grad = False

        for module in filter(lambda m: isinstance(m, nn.BatchNorm2d), self.modules()):
            module.eval()
            module.train = lambda _: None

    def unfreeze_clip(self) -> NoneType:
        for name, p in self.named_parameters():
            if not any([name.startswith(param) for param in self.TRAINABLE_PARAMS]):
                p.requires_grad = True

        for _ in filter(lambda m: isinstance(m, nn.BatchNorm2d), self.modules()):
            print("Warning this module has Batchnorm that cannot be unfrozen.")
            break

    def trainable_state_dict(self) -> Dict[str, Any]:
        return {
            k: v for k, v in self.state_dict().items() if any([k.startswith(param) for param in self.TRAINABLE_PARAMS])
        }

    def load_trainable_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> _IncompatibleKeys:
        keys = self.load_state_dict(state_dict, strict=False)
        missing_keys = [k for k in keys.missing_keys if any([k.startswith(param) for param in self.TRAINABLE_PARAMS])]
        if strict:
            error_msgs: List[str] = []
            if len(keys.unexpected_keys) > 0:
                error_msgs.insert(
                    0,
                    "Unexpected key(s) in state_dict: {}. ".format(
                        ", ".join('"{}"'.format(k) for k in keys.unexpected_keys)
                    ),
                )
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, "Missing key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in missing_keys))
                )

            if len(error_msgs) > 0:
                raise RuntimeError(
                    "Error(s) in loading state_dict for {}:\n\t{}".format(
                        self.__class__.__name__, "\n\t".join(error_msgs)
                    )
                )

        return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=keys.unexpected_keys)

    @torch.no_grad()
    def initialize_prompt(self) -> NoneType:
        if not self.learn_global_prompt and not self.learn_local_prompt:
            return

        template = self.template.replace("{}", " ").replace("_", " ").strip()
        tokenized_template = clip.tokenize(template)
        embedding = self.token_embedding(tokenized_template).type(self.dtype)
        global_prompt_init = embedding[:, 1 : 1 + self.template_init_tokens, :]

        if self.learn_global_prompt:
            self.global_prompt.data[:, : self.template_init_tokens].copy_(
                global_prompt_init.clone().expand(self.n_global_prompts, -1, -1)
            )

        if self.learn_local_prompt:
            self.local_prompts.data[:, : self.template_init_tokens].copy_(
                global_prompt_init.clone().expand(self.n_local_prompts, -1, -1)
            )

    def compute_gl_scores(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> NoneType:
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(global_logits / self.ood_temp_scale, dim=-1).cpu().numpy()
        scores = -np.max(global_probs, axis=-1)

        if local_logits is not None:
            local_probs = torch.softmax(local_logits.mean(dim=-1) / self.ood_temp_scale, dim=-1).cpu().numpy()
            local_score = -np.max(local_probs, axis=(1, 2))
            scores += local_score

        return scores

    def compute_L_mcm_scores(
        self,
        local_logits: Tensor,
    ) -> NoneType:
        assert local_logits is not None
        local_probs = torch.softmax(local_logits.mean(dim=-1) / self.ood_temp_scale, dim=-1).cpu().numpy()
        local_score = -np.max(local_probs, axis=(1, 2))
        return local_score

    def compute_mcm_scores(
        self,
        global_logits: Tensor,
    ) -> NoneType:
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(global_logits / self.ood_temp_scale, dim=-1).cpu().numpy()
        global_score = -np.max(global_probs, axis=-1)
        return global_score

    def compute_scores(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
        ood_method: Optional[str] = None,
    ) -> NoneType:
        if ood_method is None:
            ood_method = self.ood_method

        if ood_method == "GL-MCM":
            return self.compute_gl_scores(global_logits, local_logits)
        elif ood_method == "MCM":
            return self.compute_mcm_scores(global_logits)
        elif ood_method == "L-MCM":
            return self.compute_L_mcm_scores(local_logits)
        else:
            raise ValueError(f"Method {self.ood_method} not implemented")

    @torch.no_grad()
    def create_prediction_scores(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> Tensor:
        logit_scale = self.logit_scale.exp()
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(logit_scale * global_logits, dim=-1)

        if local_logits is None:
            local_probs = None
            gl_probs = global_probs
        else:
            local_logits = topk_reduce(local_logits, topk=self.topk)
            local_logits = local_logits.mean(dim=-1)
            local_probs = torch.softmax(logit_scale * local_logits, dim=-1)
            gl_logits = (global_logits + local_logits) / 2
            gl_probs = torch.softmax(logit_scale * gl_logits, dim=-1)

        return gl_probs, global_probs, local_probs
