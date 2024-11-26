import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, data=None, idx=None):
        if data is not None:
            # Use the length of the input sequence to get positional embeddings
            p = self.pe[:data].unsqueeze(0)  # Shape: [1, data, d_model]
        elif idx is not None:
            # Create positional embeddings for masked tokens
            p = self.pe[idx.view(-1)]  # Flatten idx to avoid dimension issues
            p = p.view(idx.shape[0], idx.shape[1], -1)  # Reshape to [batch_size, num_tokens, d_model]
        else:
            raise ValueError("Either data or idx must be provided.")
        return p




class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x is expected to have the shape [batch_size, length, c_in]
        x = x.permute(0, 2, 1)  # Change to [batch_size, c_in, length] for Conv1d
        x = self.tokenConv(x)  # Apply convolution
        x = x.transpose(1, 2)  # Change back to [batch_size, length, d_model]
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Ensure x has the correct shape [batch_size, length, c_in]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        elif len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(-1)  # Add batch and feature dimensions if missing

        # Make sure input channels match
        if x.shape[-1] != self.value_embedding.tokenConv.in_channels:
            raise ValueError(f"Expected input with {self.value_embedding.tokenConv.in_channels} channels, but got {x.shape[-1]} channels. Please ensure that input_c in your config matches the data features.")

        # Apply value embedding and positional embedding
        value_embed = self.value_embedding(x)
        pos_embed = self.position_embedding(value_embed.shape[1])

        x = value_embed + pos_embed
        return self.dropout(x)
