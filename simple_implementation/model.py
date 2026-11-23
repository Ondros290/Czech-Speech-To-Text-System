import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import math

from data import N_MELS, BLANK_ID


class DownSample(nn.Module):
    # downsamples by 4
    def __init__(self, n_mels, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, d, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(d, d, 3, stride=2, padding=1), nn.GELU(),
        )
 
    def forward(self, x, lens):  # x: (B, T, F)
        x = self.net(x.transpose(1, 2)).transpose(1, 2)  # (B, T/4, d)
        for _ in range(2):  # updating lengths
            lens = (lens - 1) // 2 + 1
        return x, lens
    
class DilatedResBlock(nn.Module):
    def __init__(self, d, k, dilation, p):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(d, d, k, padding=pad, dilation=dilation),
            nn.BatchNorm1d(d), 
            nn.ReLU(), 
            nn.Dropout(p),
            nn.Conv1d(d, d, 1),
            nn.BatchNorm1d(d), 
            nn.Dropout(p),
        )
        
    def forward(self, x):  # (B, d, T)
        return F.relu(x + self.net(x)) # skip connection
    
class CNN(nn.Module):
    def __init__(self, vocab, d=256, k=5, dilations=(1, 2, 4, 8, 1, 2, 4, 8), p=0.1):
        super().__init__()
        self.d_sample = DownSample(N_MELS, d)
        # doubling dilations - used for growing receptive field
        self.blocks = nn.Sequential(*[DilatedResBlock(d, k, dl, p) for dl in dilations]) 
        self.fc = nn.Linear(d, len(vocab)) # linear transform 
 
    def forward(self, x, lens):
        x, lens = self.d_sample(x, lens) # DOWNSAMPLING
        x = self.blocks(x.transpose(1, 2)).transpose(1, 2) # Dilated CONVOLUTIONS
        return self.fc(x).log_softmax(-1), lens  # LINEAR LAYER
    
class Transformer(nn.Module):
    def __init__(self, vocab, d=256, heads=4, layers=4, ff=1024, max_len=5000, dropout = 0.1):
        super().__init__()
        self.d_sample = DownSample(N_MELS, d) 
    
        pe = self.create_sinusoidal_encodings(max_len, d).unsqueeze(0) # reshape to (1, max_len, d) for batch broadcasting
        self.register_buffer("pe", pe) # non-trainable
        
        self.dropout = nn.Dropout(p = dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=ff, dropout=dropout, activation="relu", norm_first=True, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
   
        self.norm = nn.LayerNorm(d)

        self.fc = nn.Linear(d, len(vocab))
        self.softmax = nn.LogSoftmax(-1)

    def forward(self, x, lens):
 
        x, lens = self.d_sample(x, lens)
        x = x * math.sqrt(x.size(-1))
        # add positional encoding for the first T' positions
        #  + dropout
        x = x + self.pe[:, :x.size(1), :]
        x = self.dropout(x)
        
        # padding mask (B, T')
        # all sequences in a batch must be padded to the longest sequence length
        _, T, _ = x.shape
        pad_mask = torch.arange(T, device=x.device)[None, :] >= lens[:, None].to(x.device)
    
        out = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        
        out = self.norm(out)

        logits = self.fc(out)
        
        probabilities = self.softmax(logits)
        
        return probabilities, lens
        
    
    def create_sinusoidal_encodings(self, seq_len, dim):
        # this implementation was copied from: https://machinelearningmastery.com/positional-encodings-in-transformer-models/
        N = 10000
        i = torch.arange(0, dim//2)
        div_term = torch.exp(-np.log(N) * (2*i / dim))
        position = torch.arange(seq_len).unsqueeze(1)
    
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term) # Populates all even column indices with the sine of the position-frequency products
        pe[:, 1::2] = torch.cos(position * div_term) # Populates all odd column indices with the cosine of the same position-frequency products.
        return pe
    
class Decoder:
    def __init__(self, vocab):
        self.vocab = vocab
        self.blank_id = BLANK_ID
    
    def greedy_decode(self, log_probs, length):  # (T, C)
        ids = log_probs[:length].argmax(-1).tolist() # ids of the highest probabilities
        out = []
        prev= None
        for i in ids:
            if i != prev and i != 0:  # collapse repeats, then drop blanks
                out.append(self.vocab[i])
            prev = i
        return "".join(out)
    
    def decode_batch(self, log_probs, lengths):  # (N, T, C), (N,)
        decoded = []
        for k in range(log_probs.shape[0]):
            decoded.append(self.greedy_decode(log_probs[k], lengths[k].item()))
        
        return decoded

    def decode_targets(self, targets, target_lengths):
        out, offset = [], 0
        for n in target_lengths.tolist():
            ids = targets[offset : offset + n].tolist()
            chars = []
            
            for i in ids:
                char = self.vocab[i]
                chars.append(char)

            text = "".join(chars)
            
            out.append(text)
            offset += n
        return out
    
    
