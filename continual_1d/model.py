"""One shared fixed-size MLP, trained sequentially on disjoint classes."""
from torch import nn

class SequenceMLP(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, n_classes=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_classes))

    def forward(self, x):
        return self.layers(x)
