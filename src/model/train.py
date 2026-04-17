import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.model.model import MFModel
import pickle


def build_dataloader(df, batch_size):
    users_t = torch.tensor(df['user_id'].to_list(), dtype=torch.long)
    items_t = torch.tensor(df['item_id'].to_list(), dtype=torch.long)
    ratings_t = torch.tensor(df['rating'].to_list(), dtype=torch.float)
    dataset = TensorDataset(users_t, items_t, ratings_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)  # выбрасываем некратные BATCH_SIZE


def train(model, train_loader, optimizer, loss_fn, DEVICE, N_EPOCHS, model_path):
    model.to(DEVICE)
    model.train()
    print(f'start training on {DEVICE}...')

    for epoch in range(N_EPOCHS):
        total_loss = 0
        for i, (user_batch, item_batch, rating_batch) in enumerate(train_loader):
            # moving to device
            user_batch = user_batch.to(DEVICE)
            item_batch = item_batch.to(DEVICE)
            rating_batch = rating_batch.to(DEVICE)

            preds = model(user_batch, item_batch)
            loss = loss_fn(preds, rating_batch)  # ПРЯМОЙ проход

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % (len(train_loader)/100) == 0:
                print(f'batch {i} / {len(train_loader)}')
        print(f"Epoch: {epoch + 1}/{N_EPOCHS}, Loss: {total_loss}")

    torch.save(model.state_dict(), model_path)


def train_pipeline(data_path, EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, DEVICE, N_EPOCHS, model_path, mapping_path):
    df = pd.read_csv(data_path)

    with open(mapping_path, 'rb') as f:
        mappings = pickle.load(f)
    num_users = mappings["num_users"]
    num_items = mappings["num_items"]
    user_ids = mappings["user_to_id"]
    item_ids = mappings["item_to_id"]

    model = MFModel(num_users, num_items, EMBEDDING_DIM, user_ids, item_ids)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    train_loader = build_dataloader(df, BATCH_SIZE)

    train(model, train_loader, optimizer, loss_fn, DEVICE, N_EPOCHS, model_path)
