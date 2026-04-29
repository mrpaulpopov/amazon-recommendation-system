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
        user_id = user_to_id.get(user_str)

        for item in item_strs:
            item_id = item_to_id.get(item)
            if user_id is None:
                score = 0.0
                print(f"Warning: unknown user: {user_str}")
            elif item_id is None:
                score = 0.0
                print(f"Warning: unknown item: {item}")
            else:
                logit = model(
                    torch.tensor([user_id]),
                    torch.tensor([item_id])
                )
                score = logit.item()
                score = max(0.0, min(1.0, score))
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

    df = pd.read_csv(data_path)
    seen_items = set(df[df['user_id'] == user_id]['item_id'])
    all_item_ids = list(item_to_id.values())



    # --- batch inference ---
    with torch.no_grad():
        user_tensor = torch.tensor([user_id] * len(all_item_ids))
        item_tensor = torch.tensor(all_item_ids)

        scores = model(user_tensor, item_tensor)  # raw scores

    # --- convert to list ---
    scores = scores.numpy()

    item_scores = list(zip(all_item_ids, scores))



    id_to_item = {v: k for k, v in item_to_id.items()}
    seen_items_set = set(seen_items)
    item_scores = [
        {
            "item": id_to_item[i],
            "score": float(s),
            "seen": i in seen_items_set
        }
        for i, s in zip(all_item_ids, scores)
    ]

    # --- sort ---
    item_scores.sort(key=lambda x: x["score"], reverse=True)
    print(seen_items)
    print()
    print(item_scores[:20])
    print()
    print([x for x in item_scores if x["seen"]])
    print()
    print(scores.mean(), scores.std())
    return item_scores[:top_k]
