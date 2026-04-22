import pandas as pd
import pickle
from src.ml.config import Config

def preprocess(cfg: Config, samplesize=None):
    # reading from Config
    input_path = cfg.input_path
    output_path = cfg.data_path
    mapping_path = cfg.mapping_path

    df = pd.read_csv(input_path)  # item,user,rating,timestamp
    df = df.set_axis(['item', 'user', 'rating', 'timestamp'], axis=1)  # labeling

    # item, user have more than 5 reviews
    counts = df['user'].value_counts()
    valid_users = counts[counts > 5].index
    df = df[df['user'].isin(valid_users)]
    counts = df['item'].value_counts()
    valid_items = counts[counts > 5].index
    df = df[df['item'].isin(valid_items)]

    if samplesize is not None:
        df = df.sample(samplesize)

    # Как работает factorize:
    # codes, uniques = pd.factorize(np.array(["b", "b", "a", "c", "b"], dtype="O"))
    # codes: ([0, 0, 1, 2, 0])
    # uniques: (['b', 'a', 'c'])

    df['user_id'], user_uniques = pd.factorize(df['user'])
    df['item_id'], item_uniques = pd.factorize(df['item'])

    df.to_csv(output_path, index=False)

    user_to_id = {u: i for i, u in enumerate(user_uniques)}
    item_to_id = {i: j for j, i in enumerate(item_uniques)}

    id_to_user = {v: k for k, v in user_to_id.items()}
    id_to_item = {v: k for k, v in item_to_id.items()}

    # save mapping
    pickle.dump({
        "user_to_id": user_to_id,
        "item_to_id": item_to_id,
        "id_to_user": id_to_user,
        "id_to_item": id_to_item,
        "num_users": len(user_to_id),
        "num_items": len(item_to_id)
    }, open(mapping_path, "wb"))
