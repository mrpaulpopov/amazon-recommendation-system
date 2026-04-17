import torch.nn as nn

class MFModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()

        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        return (u * i).sum(dim=1)