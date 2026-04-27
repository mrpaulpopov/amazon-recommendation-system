import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.ml.model import MFModel
from src.ml.config import Config
from sklearn.model_selection import train_test_split
import pickle
import json
from torch.utils.tensorboard import SummaryWriter
import time


def build_dataloader(df, batch_size):
    users_t = torch.tensor(df['user_id'].to_list(), dtype=torch.long)
    items_t = torch.tensor(df['item_id'].to_list(), dtype=torch.long)
    ratings_t = torch.tensor(df['rating'].to_list(), dtype=torch.float)
    dataset = TensorDataset(users_t, items_t, ratings_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) # discard non-multiples of BATCH_SIZE


def train_loop(model, train_loader, val_loader, optimizer, loss_fn, N_EPOCHS, Config):
    # reading from Config
    model_path, DEVICE = Config.MODEL_PATH, Config.DEVICE
    model.to(DEVICE)
    print(f'start training on {DEVICE}...')
    # ===== TensorBoard logging =====
    writer = SummaryWriter(log_dir=f"runs/mf_model_{int(time.time())}")

    for epoch in range(N_EPOCHS):

        # learning rate decrease
        # if epoch == 4:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        #     print(f"LR reduced to {optimizer.param_groups[0]['lr']}")
        # if epoch == 40:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        #     print(f"LR reduced to {optimizer.param_groups[0]['lr']}")
        # if epoch == 60:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        #     print(f"LR reduced to {optimizer.param_groups[0]['lr']}")

        model.train()
        total_loss = 0
        for i, (user_batch, item_batch, rating_batch) in enumerate(train_loader):
            # moving to device
            user_batch = user_batch.to(DEVICE)
            item_batch = item_batch.to(DEVICE)
            rating_batch = rating_batch.to(DEVICE)

            raw_preds = model(user_batch, item_batch)
            preds = torch.clamp(raw_preds, 1.0, 5.0) # clamping
            # preds = 1 + 4 * torch.sigmoid(raw_preds) # clamping v2

            loss = loss_fn(preds, rating_batch)  # FORWARD pass

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() # sum of losses.
            train_loss = total_loss / len(train_loader) # 'normalized' loss

            # In case of big batches...
            # if i % int((len(train_loader)/4)) == 0:
            #     print(f'batch {i} / {len(train_loader)}')

        # ===== VALIDATION =====
        model.eval()
        val_loss_sum = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for user_batch, item_batch, rating_batch in val_loader:
                user_batch = user_batch.to(DEVICE)
                item_batch = item_batch.to(DEVICE)
                rating_batch = rating_batch.to(DEVICE)

                preds = model(user_batch, item_batch)
                loss = loss_fn(preds, rating_batch)

                val_loss_sum += loss.item()
                val_loss = val_loss_sum / len(val_loader)

                all_preds.append(preds.cpu())
                all_targets.append(rating_batch.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        rmse = torch.sqrt(((preds - targets) ** 2).mean())
        mae = torch.abs(preds - targets).mean()

        print(
            f"Epoch {epoch + 1}/{N_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_rmse={rmse.item():.4f} | "
            f"mae={mae.item():.4f}"
        )

        # ===== TensorBoard logging =====
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_rmse", rmse, epoch)
        writer.add_scalar("mae", mae, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    writer.close()
    torch.save(model.state_dict(), model_path)


def train_pipeline(Config, EMBEDDING_DIM, LEARNING_RATE, BATCH_SIZE, N_EPOCHS):
    # reading from Config
    data_path, model_path, mapping_path, DEVICE = Config.DATA_PATH, Config.MODEL_PATH, Config.MAPPING_PATH, Config.DEVICE

    df = pd.read_csv(data_path)

    with open(mapping_path, 'rb') as f:
        mappings = pickle.load(f)
    num_users = mappings["num_users"]
    num_items = mappings["num_items"]
    user_ids = mappings["user_to_id"]
    item_ids = mappings["item_to_id"]

    model = MFModel(num_users, num_items, EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    loss_fn = nn.MSELoss()

    # train test split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_loader = build_dataloader(train_df, BATCH_SIZE)
    val_loader = build_dataloader(val_df, BATCH_SIZE)


    train_loop(model, train_loader, val_loader, optimizer, loss_fn, N_EPOCHS, Config)

    config_json = {"EMBEDDING_DIM": EMBEDDING_DIM,
                   "BATCH_SIZE": BATCH_SIZE}
    config_path = model_path.with_name(model_path.name + ".config.json")
    with open(config_path, "w") as f:
        json.dump(config_json, f)

# tensorboard --logdir runs