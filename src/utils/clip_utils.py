import torch
from clip import clip

import src.models.clip_w_local.modified_clip_model_gallop as load_gallop_clip
import src.models.clip_w_local.modified_clip_model_locoop as load_locoop_clip
import src.models.clip_w_local.modified_clip_model_otp as load_otp_clip


def load_clip_to_cpu(clip_backbone_name, method=None):
    backbone_name = clip_backbone_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root="/path/to/clip")

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if method == "FedLoCoOp":
        model = load_locoop_clip.build_locoop_model(state_dict or model.state_dict())
    elif method == "FedOTP":
        design_details = {
            "trainer": "GLP_OT",
            "vision_depth": 0,
            "language_depth": 0,
            "vision_ctx": 0,
            "language_ctx": 0,
        }
        model = load_otp_clip.build_otp_model(state_dict or model.state_dict(), design_details)
    elif method == "FedGallop":
        model = load_gallop_clip.build_gallop_model(state_dict or model.state_dict())
    else:
        model = clip.build_model(state_dict or model.state_dict())

    if method == "FedLAPT":
        return model, clip._transform(model.visual.input_resolution)
    else:
        return model, None


class TextEncoder(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
