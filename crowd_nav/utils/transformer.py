import numpy as np
import torch
import torch.nn as nn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, source_dims, hidden_cnt):
        super(PoswiseFeedForwardNet, self).__init__()
        self.source_dims = source_dims
        self.hidden_cnt = hidden_cnt
        self.fc = nn.Sequential(
            nn.Linear(source_dims, hidden_cnt, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_cnt, source_dims, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        # print((output).shape)
        # [batch_size, seq_len, d_model]
        layernorm = nn.LayerNorm(self.source_dims)(output + residual)
        return layernorm


class ScaledDotProductAttention(nn.Module):
    def __init__(self, k_dims):
        super(ScaledDotProductAttention, self).__init__()
        self.k_dims = k_dims

    def forward(self, Q, K, V):
        # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.k_dims)
        # scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # print(attn.shape)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, source_dims, k_dims, v_dims, n_heads):
        super(MultiHeadAttention, self).__init__()
        cuda = False
        self.device = torch.device(
            "cuda:0" if cuda and torch.cuda.is_available() else "cpu")
        self.W_Q = nn.Linear(source_dims, k_dims * n_heads)
        self.W_K = nn.Linear(source_dims, k_dims * n_heads)
        self.W_V = nn.Linear(source_dims, v_dims * n_heads)
        self.fc = nn.Linear(n_heads * v_dims, source_dims)
        self.n_heads = n_heads
        self.k_dims = k_dims
        self.v_dims = v_dims
        self.source_dims = source_dims

    def forward(self, input_Q, input_K, input_V):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.k_dims).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.k_dims).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.v_dims).transpose(1, 2)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attention = ScaledDotProductAttention(k_dims=self.k_dims)(Q, K, V)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.v_dims)
        output = self.fc(context)  # [batch_size, len_q, d_model]
        # add and layer normalization
        # print(self.device)
        return nn.LayerNorm(self.source_dims).to(self.device)(output + residual), attention


class EncoderLayer(nn.Module):
    def __init__(self, source_dims, k_dims, v_dims, n_heads):
        super(EncoderLayer, self).__init__()
        self.multi_self_attention = MultiHeadAttention(source_dims, k_dims, v_dims, n_heads)
        self.feed_forward_net = PoswiseFeedForwardNet(source_dims=source_dims, hidden_cnt=100)

    def forward(self, input):
        out_put, attention = self.multi_self_attention(input, input, input)
        # print(attention.shape)
        out_put = self.feed_forward_net(out_put)
        return out_put, attention


class TransformerEncoder(nn.Module):
    def __init__(self, source_dims, k_dims, v_dims, n_heads, layer_cnt=1):
        super(TransformerEncoder, self).__init__()
        cuda = False
        self.device = torch.device(
            "cuda:0" if cuda and torch.cuda.is_available() else "cpu")
        self.layers = nn.ModuleList(
            [EncoderLayer(source_dims, k_dims, v_dims, n_heads) for _ in range(layer_cnt)])

    def forward(self, input):
        cnt = 0
        enc_output = input
        for layer in self.layers:
            enc_output, res_attention = layer(enc_output)
            cnt += 1

        return enc_output
