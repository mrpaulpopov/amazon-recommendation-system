import torch
import pickle
from src.ml.model import MFModel
from src.ml.config import Config
import json

def predict_user_items(cfg: Config, user_str, item_strs):
    # reading from Config
    model_path, mapping_path = cfg.model_path, cfg.mapping_path

    with open(mapping_path, "rb") as f:
        mappings = pickle.load(f)
    user_to_id = mappings["user_to_id"]
    item_to_id = mappings["item_to_id"]
    num_users = mappings["num_users"]
    num_items = mappings["num_items"]

    # Считываем Embedding Dim из сохраненного конфига. Это значение должно не меняться между train и predict.
    config_path = model_path.with_name(model_path.name + ".config.json")
    with open(config_path) as f:
        json_data = json.load(f)
    embedding_dim = json_data["EMBEDDING_DIM"]

    model = MFModel(num_users, num_items, embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        # считаем средние embedding
        mean_user_emb = model.user_emb.weight.mean(dim=0)
        mean_item_emb = model.item_emb.weight.mean(dim=0)

        # --- user ---
        if user_str in user_to_id:
            user_id = user_to_id[user_str]
            user_vec = model.user_emb(torch.tensor(user_id))
        else:
            print(f"Unknown user: {user_str} -> using mean embedding")
            user_vec = mean_user_emb

        preds = []

        # --- items ---
        for item in item_strs:
            if item in item_to_id:
                item_id = item_to_id[item]
                item_vec = model.item_emb(torch.tensor(item_id))
            else:
                print(f"Unknown item: {item} -> using mean embedding")
                item_vec = mean_item_emb

            score = torch.sigmoid((user_vec * item_vec).sum())
            preds.append(score)

        return torch.stack(preds).detach().cpu().numpy().tolist() if preds else 0