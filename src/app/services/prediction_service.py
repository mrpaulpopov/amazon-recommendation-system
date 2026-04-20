from src.ml.predict import predict_user_items

def predict_service(Config, user_str, item_strs):
    return predict_user_items(Config, user_str, item_strs)