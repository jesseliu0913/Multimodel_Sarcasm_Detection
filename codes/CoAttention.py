from typing import Dict, Optional

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q)  # batch_size x len_q x len_k


def masked_softmax(scores, src_length_masking=True):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""
    if src_length_masking:
        batch_size, max_src_len = scores.size()
        # compute masks
        src_mask = get_attn_pad_mask(scores, scores)
        # Fill pad positions with -inf
        scores = scores.masked_fill(src_mask == 0, -1e9)

    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)


class ParallelCoAttentionNetwork(nn.Module):

    def __init__(self, hidden_dim, co_attention_dim, src_length_masking=True):
        super(ParallelCoAttentionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.co_attention_dim = co_attention_dim
        self.src_length_masking = src_length_masking

        self.W_b = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
        self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))

    def forward(self, V, Q):
        """
        :param V: batch_size * hidden_dim * region_num, eg B x 512 x 196
        :param Q: batch_size * seq_len * hidden_dim, eg B x L x 512
        :param Q_lengths: batch_size
        :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
        batch_size * hidden_dim, batch_size * hidden_dim
        """
        # (batch_size, seq_len, region_num)
        C = torch.matmul(Q, torch.matmul(self.W_b, V))
        # (batch_size, co_attention_dim, region_num)
        H_v = nn.Tanh()(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
        # (batch_size, co_attention_dim, seq_len)
        H_q = nn.Tanh()(
            torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))

        # (batch_size, 1, region_num)
        a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)
        # (batch_size, 1, seq_len)
        a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)
        # # (batch_size, 1, seq_len)

        masked_a_q = masked_softmax(
            a_q.squeeze(1), self.src_length_masking
        ).unsqueeze(1)

        # (batch_size, hidden_dim)
        v = torch.matmul(a_v, V.permute(0, 2, 1)).squeeze(1)
        # (batch_size, hidden_dim)
        q = torch.matmul(masked_a_q, Q).squeeze(1)
        # print(q.shape)
        # print(v.shape)

        return a_v, masked_a_q, v, q

#
# co_atten = ParallelCoAttentionNetwork(768, 2)
# v = torch.randn(2, 768, 196)
# q = torch.randn(2, 196, 768)
# a_v, a_q, v, q = co_atten(v, q)
# print(a_v)
# print(a_v.shape)
# print(a_q)
# print(a_q.shape)
# print(v)
# print(v.shape)
# print(q)
# print(q.shape)