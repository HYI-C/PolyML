import torch
import torch.nn.functional as F
from torch import nn

class Encoder(nn.Module):
    '''This is the encoder, the decoder are the heads'''
    def __init__(
        self, 
        emb: None, # this is the embedder
        rnn_dim: int,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.emb = emb
        self.rnn = nn.GRU(
            input_size = emb.output_shape[1], 
            hidden_size = rnn_dim,
            num_layers = 1,
            dropout = drop_rate,
            batch_first = True,
        )
        self.linear = nn.Linear(rnn_dim, rnn_dim)
        self.output_shape = [None, rnn_dim]

    def forward(self, x):
        x_emb = self.emb(x)
        #..sequence inds = zip(x["offsets"][:-1, x["offsets"][1:]]) where
        #offsets are the dates
        x_stacked = nn.utils.rnn.pack_sequence(
            [x_emb[s:e] for s, e in zip(x["offsets"][:-1], x["offsets"][1:])],
            enforce_sorted=False,
        )
        x_proc, _ = self.rnn(x_stacked)
        padded, lens = nn.utils.rnn.pad_packed_sequence(x_proc)
        seq = torch.cat([padded[:seqlen, i] for i, seqlen in enumerate(lens)])
        assert not torch.isnan(seq.data.mean()), "NaN value in rnn output"

        x_out = self.linear(F.relu(F.dropout(seq)))

        return x_out

