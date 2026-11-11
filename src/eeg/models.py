"""
Neural network architectures for EEG encoding (1D-CNN + BiLSTM).
"""
from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn

N_CHANNELS: int = 23
EMBED_DIM: int = 128


class EEGEncoder(nn.Module):
    """1D-CNN + BiLSTM EEG encoder with a classification head and a bottleneck embedding.

    Input  x     : (batch, 23, 1024)
    Output logit : (batch,)        raw seizure logit (use BCEWithLogitsLoss / sigmoid)
           z_eeg : (batch, 128)    bottleneck embedding used later for retrieval
    """

    def __init__(
        self,
        in_channels: int = N_CHANNELS,
        embed_dim: int = EMBED_DIM,
        cnn_channels: Tuple[int, ...] = (64, 128, 128),
        kernel_sizes: Tuple[int, ...] = (7, 5, 3),
        lstm_hidden: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        prev = in_channels
        for out, k in zip(cnn_channels, kernel_sizes):
            blocks += [
                nn.Conv1d(prev, out, kernel_size=k, padding=k // 2),  # mixes channels + temporal filters
                nn.BatchNorm1d(out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),                                      # halves the time axis
            ]
            prev = out
        self.cnn = nn.Sequential(*blocks)

        self.bilstm = nn.LSTM(
            input_size=prev, hidden_size=lstm_hidden,
            batch_first=True, bidirectional=True,
        )
        lstm_out = 2 * lstm_hidden  # 128 (forward + backward)

        # mean-pool + max-pool over time -> 2 * lstm_out -> bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(2 * lstm_out, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, 23, 1024)
        h = self.cnn(x)                       # (batch, 128, 128)  [channels, time] after 3 pools: 1024 -> 128
        h = h.transpose(1, 2)                 # (batch, 128, 128)  [time, features] for the LSTM
        h, _ = self.bilstm(h)                 # (batch, 128, 128)  [time, 2 * lstm_hidden]
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)  # (batch, 256)
        z_eeg = self.bottleneck(pooled)       # (batch, 128)
        logit = self.classifier(z_eeg).squeeze(-1)  # (batch,)
        return logit, z_eeg
