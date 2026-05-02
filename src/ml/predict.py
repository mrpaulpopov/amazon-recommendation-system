import torch
import pickle
from src.ml.model import MFModel
from src.ml.config import Config
import json
import pandas as pd


def predict_user_items(Config, user_str, item_strs):
    # reading from Config
    model_path, mapping_path = Config.MODEL_PATH, Config.MAPPING_PATH

    with open(mapping_path, "rb") as f:
        mappings = pickle.load(f)
    user_to_id = mappings["user_to_id"]
    item_to_id = mappings["item_to_id"]
    num_users = mappings["num_users"]
    num_items = mappings["num_items"]

    # Reading EMBEDDING DIM value from json. This value should not be changed between train and predict.
    config_path = model_path.with_name(model_path.name + ".config.json")
    with open(config_path) as f:
        json_data = json.load(f)
    embedding_dim = json_data["EMBEDDING_DIM"]

    model = MFModel(num_users, num_items, embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        preds = []
        user_id = user_to_id.get(user_str)  # user_str -> user_id

        for item in item_strs:
            item_id = item_to_id.get(item)  # item -> item_id
            if user_id is None:
                score = 0.0
                print(f"Warning: unknown user: {user_str}")
            elif item_id is None:
                score = 0.0
                print(f"Warning: unknown item: {item}")
            else:
                # this is logit (model returns prediction if 'return torch.sigmoid(...)'
                logit = model(torch.tensor([user_id]), torch.tensor([item_id]))
                score = logit.item()  # score = logits, it's not probability.
            preds.append(score)

        return preds if preds else 0


def recommend_for_user(Config, user_str, top_k):
    data_path, model_path, mapping_path = Config.DATA_PATH, Config.MODEL_PATH, Config.MAPPING_PATH

    # --- load mappings ---
    with open(mapping_path, "rb") as f:
        mappings = pickle.load(f)

    user_to_id = mappings["user_to_id"]
    item_to_id = mappings["item_to_id"]
    num_users = mappings["num_users"]
    num_items = mappings["num_items"]

    user_id = user_to_id.get(user_str)
    if user_id is None:
        raise ValueError(f"Unknown user: {user_str}")

    # --- load model config ---
    config_path = model_path.with_name(model_path.name + ".config.json")
    with open(config_path) as f:
        json_data = json.load(f)

    embedding_dim = json_data["EMBEDDING_DIM"]

    # --- load model ---
    model = MFModel(num_users, num_items, embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # --- seen items ---
    # get previously seen item IDs for a given user
    df = pd.read_csv(data_path)
    mask = df['user_id'] == user_id  # pandas boolean mask
    filtered_df = df[mask]  # filter rows using mask
    items_series = filtered_df['item_id']  # SELECT 'item_id' column
    seen_items_set = set(items_series)  # pandas.Series to set

    all_item_ids = list(item_to_id.values())

    # --- batch inference ---
    with torch.no_grad():
        item_tensor = torch.tensor(all_item_ids)  # tensor of all item ids
        user_tensor = torch.tensor(
            [user_id] * len(all_item_ids))  # item_t = [i1, i2, i3, i4..], user_t = [u, u, u, u..]

        scores = model(user_tensor, item_tensor)  # raw scores

    id_to_item = {v: k for k, v in item_to_id.items()}  # dict comprehension: swapping

    # list of dictionaries (each item has multiple fields)
    item_scores = []
    for i, s in zip(all_item_ids, scores):  # iterate over item ids and model scores in parallel
        item_scores.append({
            "item": id_to_item[i],  # item string
            "score": float(s),  # tensor to float
            "seen": i in seen_items_set  # is seen?
        })

    # --- sort ---
    item_scores.sort(key=lambda x: x["score"], reverse=True)
    return item_scores[:top_k]
