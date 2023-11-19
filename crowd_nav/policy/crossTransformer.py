import torch
import torch.nn as nn

import numpy as np
from einops import rearrange
from torch import einsum
from torch.nn import functional as F


class self_Attention(nn.Module):
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
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, n_mask=None):
        B, n_heads, len1, len2, d_k = Q.shape

        # print(B, n_heads, len1, len2, d_k)
        # print(Q.shape, K.transpose(-1, -2).shape)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # print(scores.shape)
        if n_mask is not None:
            n_mask = n_mask.to(scores.device)
            n_mask = n_mask.unsqueeze(1).repeat(1, 4, 1, 1, 1)
            scores.masked_fill_(n_mask, -1e9)
        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, V)

        return context


class SMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V, spatial_n_mask=None):
        B, N, T, C = input_Q.shape
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).transpose(1, 3)

        context = ScaledDotProductAttention()(Q, K, V, spatial_n_mask)
        context = context.permute(0, 3, 2, 1, 4)
        context = context.reshape(B, N, T, self.heads * self.head_dim)

        output = self.fc_out(context)
        return output


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V, temporal_n_mask=None):
        B1, T1, N1, C1 = input_Q.shape
        B2, T2, N2, C2 = input_K.shape
        # B, T, N, C = input_V.shape
        # print(B2, T2, N2, C2)
        Q = self.W_Q(input_Q).view(B1, N1, T1, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        K = self.W_K(input_K).view(B2, N2, T2, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = self.W_V(input_V).view(B2, N2, T2, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        # print(K.shape)

        context = ScaledDotProductAttention()(Q, K, V, temporal_n_mask)
        context = context.permute(0, 2, 3, 1, 4)

        context = context.reshape(B1, T1, N1, self.heads * self.head_dim)
        # print(context.shape)
        #
        output = self.fc_out(context)
        return output


class TTransformer(nn.Module):
    def __init__(self, embed_size, heads, time_num, device, dropout=0., forward_expansion=1):
        super(TTransformer, self).__init__()

        self.time_num = time_num
        self.temporal_embedding = nn.Embedding(time_num, embed_size)
        self.attention = TMultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, query, key, value, pe=False, temporal_n_mask=None):
        B, T, N, C = query.shape

        if pe:
            D_T = self.temporal_embedding(torch.arange(0, T).to(self.device))
            D_T = D_T.expand(B, N, T, C)
            query = query + D_T
        attention = self.attention(query, key, value, temporal_n_mask)
        # print(attention.shape, query.shape)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out
