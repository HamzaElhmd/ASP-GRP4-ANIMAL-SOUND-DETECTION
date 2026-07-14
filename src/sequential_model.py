import torch
import torch.nn as nn


class SequentialEventDetector(nn.Module):
    """GRU/LSTM over a mel-spectrogram time sequence, independent sigmoid
    per class per timestep (not softmax) so overlapping animals aren't
    structurally excluded. Outputs raw logits -- pair with
    BCEWithLogitsLoss for training, torch.sigmoid(...) for inference."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 5,
        rnn_type: str = "gru",
        dropout: float = 0.0,
    ):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.output_layer(out)  # (batch, time, num_classes) logits
