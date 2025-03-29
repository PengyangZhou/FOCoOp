import math
import os

import clip
import torch
import torch.nn as nn

from src.utils.clip_utils import TextEncoder


class CLIP_scoring(nn.Module):
    def __init__(self, clip_model, tokenized_prompts):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.tokenized_prompts = tokenized_prompts
        self.token_embedding = clip_model.token_embedding

    def prepare_id(self, tokenized_prompts_id):
        with torch.no_grad():
            prompts = self.token_embedding(tokenized_prompts_id).type(self.dtype)
            text_features = self.text_encoder(prompts, tokenized_prompts_id)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features
        cos_sim = text_features @ self.text_features.t()
        return cos_sim

    def forward(self, tokenized_prompts_wordnet):
        with torch.no_grad():
            prompts = self.token_embedding(tokenized_prompts_wordnet).type(self.dtype)
            text_features_wordnet = self.text_encoder(prompts, tokenized_prompts_wordnet)

            text_features_wordnet = text_features_wordnet / text_features_wordnet.norm(dim=-1, keepdim=True)

            cos_sim = text_features_wordnet @ self.text_features.t()

        return cos_sim


def Deduplication(listone, listtwo):
    new_list = []
    for item in listone:
        if item in listtwo:
            pass
        else:
            new_list.append(item)
    return new_list


def generate_cossim_idname_wordnet_dedup(classnames, save_path, clip_model=None):
    print("starting generating cosine similarity with ID classnames and wordnet names..........")

    current_dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(current_dir, "../txtfiles")
    prompts = ["The nice " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

    text_files = os.listdir(path)
    adj_list = []
    noun_list = []

    for text_file in text_files:
        if text_file[:3] == "adj" and text_file[-3:] == "txt":
            file_path = os.path.join(path, text_file)
            try:
                with open(file_path, "r") as file:
                    text = file.read()
            except FileNotFoundError:
                print(f"The file {file_path} was not found. Please make sure the file path is correct.")
            adj_list += text.split("\n")[:-1]
        elif text_file[:4] == "noun" and text_file[-3:] == "txt":
            file_path = os.path.join(path, text_file)
            try:
                with open(file_path, "r") as file:
                    text = file.read()
            except FileNotFoundError:
                print(f"The file {file_path} was not found. Please make sure the file path is correct.")
            noun_list += text.split("\n")[:-1]
        else:
            print("skip a not text file", text_file)
    adj_list = list(set(adj_list))
    noun_list = list(set(noun_list))
    print("length of adj words", len(adj_list))
    print("length of noun words", len(noun_list))

    adj_list = Deduplication(adj_list, classnames)
    noun_list = Deduplication(noun_list, classnames)
    print("length of adj words after image dedup", len(adj_list))
    print("length of noun words after image dedup", len(noun_list))
    adj_list = Deduplication(adj_list, noun_list)
    print("length of adj words after noun dedup", len(adj_list))

    prompts_noun = ["The nice " + name + "." for name in noun_list]
    tokenized_prompts_noun = torch.cat([clip.tokenize(p) for p in prompts_noun]).cuda()

    prompts_adj = ["This is a " + name + " photo." for name in adj_list]
    tokenized_prompts_adj = torch.cat([clip.tokenize(p) for p in prompts_adj]).cuda()

    if clip_model is not None:
        model, _ = clip.load("ViT-B/16", "cuda", jit=False)
    else:
        model = clip_model

    clip_selector = CLIP_scoring(model, tokenized_prompts)
    cos_sim_id = clip_selector.prepare_id(tokenized_prompts)
    batch_size = 100

    total_iter = math.ceil(len(adj_list) / batch_size)
    can_list_adj = []
    can_cos_adj = []
    for i in range(total_iter):
        if i % 10 == 0:
            print(i, total_iter)
        if i == total_iter - 1:
            current_list = adj_list[i * batch_size :]
            current_token = tokenized_prompts_adj[i * batch_size :, :]
        else:
            current_list = adj_list[i * batch_size : (i + 1) * batch_size]
            current_token = tokenized_prompts_adj[i * batch_size : (i + 1) * batch_size]
        cos_sim = clip_selector(current_token)
        can_list_adj += current_list
        can_cos_adj.append(cos_sim)
    can_cos_adj = torch.cat(can_cos_adj, dim=0)

    total_iter = math.ceil(len(noun_list) / batch_size)
    can_list_noun = []
    can_cos_noun = []
    for i in range(total_iter):
        if i % 10 == 0:
            print(i, total_iter)
        if i == total_iter - 1:
            current_list = noun_list[i * batch_size :]
            current_token = tokenized_prompts_noun[i * batch_size :, :]
        else:
            current_list = noun_list[i * batch_size : (i + 1) * batch_size]
            current_token = tokenized_prompts_noun[i * batch_size : (i + 1) * batch_size]
        cos_sim = clip_selector(current_token)
        can_list_noun += current_list
        can_cos_noun.append(cos_sim)
    can_cos_noun = torch.cat(can_cos_noun, dim=0)

    save_dict = {}
    save_dict["text_list_adj"] = can_list_adj
    save_dict["cos_sim_adj"] = can_cos_adj.cpu()

    save_dict["text_list_noun"] = can_list_noun
    save_dict["cos_sim_noun"] = can_cos_noun.cpu()

    save_dict["cos_sim_id"] = cos_sim_id.cpu()

    torch.save(save_dict, save_path)
