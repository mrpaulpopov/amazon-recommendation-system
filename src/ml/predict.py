import torch
import pickle
from src.ml.model import MFModel
from src.ml.config import Config
import json

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
                score = torch.sigmoid(logit).item()
            preds.append(score)

        return preds if preds else 0