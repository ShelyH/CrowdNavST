import math

import torch
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
from torch.autograd import Variable


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(CrossAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = torch.softmax(att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # print(x.shape)
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, d_model, seq_length, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.dim = d_model
        self.layers = nn.ModuleList([])
        self.position_embedding = PositionalEncoding(d_model=d_model)
        self.embedding = nn.Embedding(5, d_model)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(d_model, Attention(d_model, heads=heads, dim_head=dim_head, dropout=0.))),
                Residual(PreNorm(d_model, FeedForward(d_model, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, pe=False):
        if pe:
            # x = self.position_embedding(x)
            D_T = self.embedding(torch.arange(0, 5).to("cuda:0"))
            # print(x.shape)
            D_T = D_T.expand(x.shape[0], x.shape[1], x.shape[2])
            x = x + D_T
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x

    @staticmethod
    def positional_encoding(seq_length, d_model):
        # 生成位置编码张量
        pos_enc = torch.zeros(seq_length, d_model)
        for pos in range(seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pos_enc[pos, i] = math.sin(pos / 10000 ** (2 * i / d_model))
                else:
                    pos_enc[pos, i] = math.cos(pos / 10000 ** ((2 * i - 1) / d_model))
        return pos_enc


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:, :x.size(1)]
        x = x + Variable(pe, requires_grad=False)
        return self.dropout(x)

# def create_sinusoidal_embeddings(nb_p, dim, E):
#     theta = np.array([
#         [p / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
#         for p in range(nb_p)
#     ])
#     with torch.no_grad():
#         E[:, 0::2] = torch.FloatTensor(np.sin(theta[:, 0::2]))
#         E[:, 1::2] = torch.FloatTensor(np.cos(theta[:, 1::2]))
#     E.requires_grad = False

# class Embeddings(nn.Module):
#     def __init__(self, d_model, max_position_embeddings):
#         super().__init__()
#         self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)  # Embedding(10000, 32)
#         create_sinusoidal_embeddings(
#             nb_p=max_position_embeddings,
#             dim=d_model,
#             E=self.position_embeddings.weight
#         )
#
#         self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
#
#     def forward(self, input):
#         input_ids = input[:, :, 0]
#         seq_length = input_ids.size(1)  # time_seq
#         position_ids = torch.arange(seq_length, dtype=torch.long, device=input.device)  # (max_seq_length)
#         position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)
#
#         # Get position embeddings for each position id
#         position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
#         # print(position_embeddings)
#         # Add them both
#         embeddings = input + position_embeddings  # (bs, max_seq_length, dim)
#
#         return embeddings
