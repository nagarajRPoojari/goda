import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dtype: torch.dtype):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc2 = nn.Linear(embed_dim, hidden_dim, dtype=dtype, bias=False)
        self.fc3 = nn.Linear(hidden_dim, embed_dim, dtype=dtype, bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.gelu(x_fc1, approximate="tanh") * x_fc2
        return self.fc3(x)
