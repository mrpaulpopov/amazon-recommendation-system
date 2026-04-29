import torch.nn as nn
import torch

class MFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        # bias
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        dot = (u * i).sum(dim=1)
        return (dot + self.user_bias(user_ids).squeeze(-1)
                + self.item_bias(item_ids).squeeze(-1)
                + self.global_bias.squeeze(-1))
        # return dot # without bias