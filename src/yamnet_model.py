import torch
import torch.nn as nn

try:
    from yamnet_features import EMBEDDING_DIM
except ModuleNotFoundError:
    from src.yamnet_features import EMBEDDING_DIM


class YamnetClassifierHead(nn.Module):
    """Small trainable head on top of frozen YAMNet embeddings -- YAMNet
    itself is never updated. Independent sigmoid per class per timestep
    (not softmax), same multi-label design as every other track in this
    project. Applied independently per timestep (each YAMNet embedding
    already summarizes ~0.96s of audio context, so there's no need for an
    extra recurrent layer here -- that's the sequential track's job)."""

    def __init__(self, input_size: int = EMBEDDING_DIM, hidden_size: int = 128, num_classes: int = 5, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (batch, time, num_classes) logits
