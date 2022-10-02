# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import math
# import pickle
# import os
# import torch.utils.data as Data
# from transformers import BertTokenizer, BertModel, BertForSequenceClassification
# import LoadData
# from  AttributeFeature import ExtractAttributeFeature
# import Config
# from transformers import logging
# import torch.nn.functional as F
# logging.set_verbosity_error()
#
#
# device = Config.device
# tokenizer = Config.tokenizer
#
# WORKING_PATH = Config.WORKING_PATH
#
#
# d_model = Config.d_model
# d_ff = Config.d_ff
# d_k = d_v = Config.d_v
# n_layers = Config.n_layers
# n_heads = Config.n_heads
#
#
# def get_attn_pad_mask(seq_q, seq_k):
#     batch_size, len_q = seq_q.size()
#     batch_size, len_k = seq_k.size()
#     # eq(zero) is PAD token
#     pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
#     return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k
#
#
# class EmbeddingModule(nn.Module):
#     def __init__(self):
#         super(EmbeddingModule, self).__init__()
#         self.bert = BertModel.from_pretrained('../model').to(device)
#
#     def forward(self, input_ids, atttention_mask, token_type_ids):
#         input_ids, atttention_mask, token_type_ids = input_ids.to(device), atttention_mask.to(device), token_type_ids.to(device)
#         word_embed = self.bert(input_ids, atttention_mask, token_type_ids).last_hidden_state
#         return word_embed
#
#
# class ScaledDotProductAttention(nn.Module):
#     def __init__(self):
#         super(ScaledDotProductAttention, self).__init__()
#
#     def forward(self, Q, K, V, attn_mask):
#         # input: [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]
#         # V: [batch_size x n_heads x len_k x d_v]
#         scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)   # output: [batch_size x n_heads x len_q x len_k]
#         # print("score_shape:", scores.shape)  # [32, 1, 75, 75]
#         # print("attn_mask", attn_mask.shape)  # [32, 1, 75, 75]
#
#         scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
#         attn = nn.Softmax(dim=-1)(scores)
#         context = torch.matmul(attn, V)
#         return context, attn
#
# # wo
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self):
#         super(MultiHeadAttention, self).__init__()
#
#         self.W_Q = nn.Linear(d_model, d_k * n_heads).to(device)
#         self.W_K = nn.Linear(d_model, d_k * n_heads).to(device)
#         self.W_V = nn.Linear(d_model, d_v * n_heads).to(device)
#         self.linear = nn.Linear(n_heads * d_v, d_model).to(device)
#         self.layer_norm = nn.LayerNorm(d_model).to(device)
#
#     def forward(self, Q, K, V, attn_mask):
#         # print(attn_mask.shape)
#         # input: Q: [batch_size x len_q x d_moel], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
#         residual, batch_size = Q, Q.size(0)
#         # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
#
#         q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
#         # print(q_s.shape)  # [32, 2, 75, 64]
#         k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
#         v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]
#
#         # input: batch_size x len_q x len_k
#         # output: attn_mask : [batch_size x n_heads x len_q x len_k]
#         # print(attn_mask.shape)  # [32, 75, 75]
#         attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
#         # print(attn_mask.shape)  # [batch_size, n_heads, seq_length, seq_length]
#
#         # output：context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
#         context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
#         context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
#         output = self.linear(context)
#         return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]
#
#
#
# class PoswiseFeedForwardNet(nn.Module):
#     def __init__(self):
#         super(PoswiseFeedForwardNet, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1).to(device)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1).to(device)
#         self.layer_norm = nn.LayerNorm(d_model).to(device)
#         self.act = nn.ReLU().to(device)
#
#     def forward(self, enc_inputs, enc_outputs):
#         residual = enc_inputs # inputs : [batch_size, len_q, d_model]
#         output = self.act(self.conv1(enc_outputs.transpose(1, 2)))
#         output1 = self.conv2(output).transpose(1, 2)
#         return self.layer_norm(output1 + residual)
#
#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model).to(device)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)# even
#         pe[:, 1::2] = torch.cos(position * div_term)# odd
#         # output: pe:[max_len*d_model]
#
#         # output：[max_len*1*d_model]
#         pe = pe.unsqueeze(0).transpose(0, 1)
#
#         self.register_buffer('pe', pe)  # 定一个缓冲区，其实简单理解为这个参数不更新就可以
#
#     def forward(self, x):
#         """
#         x: [seq_len, batch_size, d_model]
#         """
#         y = x + self.pe[:x.size(0), :]
#         return self.dropout(y)
#
#
#
# class EncoderLayer(nn.Module):
#     def __init__(self):
#         super(EncoderLayer, self).__init__()
#         self.enc_self_attn = MultiHeadAttention()
#         self.label_enc_attn = MultiHeadAttention()
#         self.pos_ffn = PoswiseFeedForwardNet()
#
#     def forward(self, enc_inputs, enc_self_attn_mask, label_embeded, label_enc_attn_mask):
#         # enc_inputs：[batch_size x seq_len_q x d_model]
#         enc_inputs, enc_self_attn_mask, label_embeded, label_enc_attn_mask = enc_inputs.to(device), enc_self_attn_mask.to(device), label_embeded.to(device), label_enc_attn_mask.to(device)
#         enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
#         enc_outputs, label_enc_attn = self.label_enc_attn(label_embeded, enc_outputs, enc_outputs, label_enc_attn_mask)
#         enc_outputs = self.pos_ffn(enc_inputs, enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
#         return enc_outputs, attn, label_enc_attn
#
#
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.src_emb = EmbeddingModule()
#         self.pos_emb = PositionalEncoding(d_model)
#         self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
#
#     def forward(self, input_ids, atttention_mask, token_type_ids, label_embeded, label_input):
#         # input_ids: [batch_size, seq_len]
#
#         # enc_outputs: .[batch_size, seq_len, d_model]
#
#         enc_outputs = self.src_emb(input_ids, atttention_mask, token_type_ids)
#
#         enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
#
#         enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
#         # print(enc_self_attn_mask)
#         label_enc_attn_mask = get_attn_pad_mask(input_ids, label_input)
#
#         enc_self_attns, label_enc_attns = [], []
#         for layer in self.layers:
#             enc_outputs, enc_self_attn, label_enc_attn = layer(enc_outputs, enc_self_attn_mask, label_embeded, label_enc_attn_mask)
#             label_enc_attns.append(label_enc_attn)
#             enc_self_attns.append(enc_self_attn)
#         return enc_outputs, enc_self_attns, label_enc_attns
#
#
#
# class Transformer(nn.Module):
#     def __init__(self, input_ids, atttention_mask, token_type_ids, label_embeded, label_input):
#         super(Transformer, self).__init__()
#         self.encoder = Encoder()
#         self.input_ids = input_ids
#         self.atttention_mask = atttention_mask
#         self.token_type_ids = token_type_ids
#         self.label_embeded = label_embeded
#         self.label_input = label_input
#         self.fc = nn.Linear(768, 2)
#
#     def forward(self):
#
#         enc_outputs, enc_self_attns, label_enc_attns = self.encoder(self.input_ids, self.atttention_mask, self.token_type_ids, self.label_embeded, self.label_input)
#
#         # enc_outputs = enc_outputs.mean(dim=1)
#         # enc_outputs = self.fc(enc_outputs)
#         return enc_outputs, enc_self_attns, label_enc_attns
#
#
#
#
#
# if __name__ == '__main__':
#     # pass
#
#     test = ExtractAttributeFeature()
#
#
#     for input_ids, atten_mask, image_feature, token_type_ids, label, group, id in LoadData.train_loader:
#         input_ids, atten_mask, image_feature, token_type_ids, label = input_ids.to(device), atten_mask.to(device), \
#                                                                       image_feature.to(device), token_type_ids.to(
#             device), label
#         label_embeded, label_input = test(label)  # [batch_size, seq_length, hidden_size]
#         model = Transformer(input_ids, atten_mask, token_type_ids, label_embeded, label_input)
#         enc_outputs, enc_self_attns, label_enc_attns = model()
#         print(enc_outputs.shape)
# #         loss_fn = nn.CrossEntropyLoss()
# #         optimizer = optim.Adam(model.parameters(), lr=1e-5)
# #
# #         optimizer.zero_grad()
# #         print(group.shape)
# #         loss = loss_fn(enc_outputs, group)
# #         print("损失：", loss.item())
# #         # print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
# #         loss.backward(retain_graph=True)
# #         optimizer.step()
# #
# #
# #
# #../model
#
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import pickle
import os
import torch.utils.data as Data
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import LoadData
from  AttributeFeature import ExtractAttributeFeature
import Config
from transformers import logging
import torch.nn.functional as F
logging.set_verbosity_error()


device = Config.device
tokenizer = Config.tokenizer

WORKING_PATH = Config.WORKING_PATH


d_model = Config.d_model
d_ff = Config.d_ff
d_k = d_v = Config.d_v
n_layers = Config.n_layers
n_heads = Config.n_heads


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class EmbeddingModule(nn.Module):
    def __init__(self):
        super(EmbeddingModule, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        # self.fc1 = nn.Linear(768, 128).to(device)

    def forward(self, input_ids, atttention_mask, token_type_ids):
        word_embed = self.bert(input_ids, atttention_mask, token_type_ids).last_hidden_state
        # word_embed = self.fc1(word_embed)
        return word_embed


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # input: [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]
        # V: [batch_size x n_heads x len_k x d_v]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)   # output: [batch_size x n_heads x len_q x len_k]
        # print("score_shape:", scores.shape)  # [32, 1, 75, 75]
        # print("attn_mask", attn_mask.shape)  # [32, 1, 75, 75]

        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

# wo

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model).to(device)
        self.layer_norm = nn.LayerNorm(d_model).to(device)

    def forward(self, Q, K, V, attn_mask):
        # print(attn_mask.shape)
        # input: Q: [batch_size x len_q x d_moel], K: [batch_size x len_k x d_model], V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # print("Q",Q.shape)

        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        # print(q_s.shape)  # [32, 2, 75, 64]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # input: batch_size x len_q x len_k
        # output: attn_mask : [batch_size x n_heads x len_q x len_k]
        # print(attn_mask.shape)  # [32, 75, 75]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # print(attn_mask.shape)  # [batch_size, n_heads, seq_length, seq_length]

        # output：context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1).to(device)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1).to(device)
        self.layer_norm = nn.LayerNorm(d_model).to(device)
        self.act = nn.ReLU().to(device)

    def forward(self, enc_inputs, enc_outputs):
        residual = enc_inputs # inputs : [batch_size, len_q, d_model]
        output = self.act(self.conv1(enc_outputs.transpose(1, 2)))
        output1 = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output1 + residual)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout).to(device)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)# even
        pe[:, 1::2] = torch.cos(position * div_term)# odd
        # output: pe:[max_len*d_model]

        # output：[max_len*1*d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  # 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        # print(x.shape)
        y = x + self.pe[:x.size(0), :]

        return self.dropout(y)



class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention().to(device)
        self.label_enc_attn = MultiHeadAttention().to(device)
        self.pos_ffn = PoswiseFeedForwardNet().to(device)

    def forward(self, enc_inputs, enc_self_attn_mask, label_embeded, label_enc_attn_mask):
        # enc_inputs：[batch_size x seq_len_q x d_model]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs, label_enc_attn = self.label_enc_attn(label_embeded, enc_outputs, enc_outputs, label_enc_attn_mask)
        enc_outputs = self.pos_ffn(enc_inputs, enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn, label_enc_attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = EmbeddingModule().to(device)
        self.pos_emb = PositionalEncoding(d_model).to(device)
        self.layers = nn.ModuleList([EncoderLayer().to(device) for _ in range(n_layers)])

    def forward(self, input_ids, atttention_mask, token_type_ids, label_embeded, label_input):
        # input_ids: [batch_size, seq_len]

        # enc_outputs: .[batch_size, seq_len, d_model]

        enc_outputs = self.src_emb(input_ids, atttention_mask, token_type_ids)

        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        # print(enc_self_attn_mask)
        label_enc_attn_mask = get_attn_pad_mask(input_ids, label_input)

        enc_self_attns, label_enc_attns = [], []
        for layer in self.layers:
            enc_outputs, enc_self_attn, label_enc_attn = layer(enc_outputs, enc_self_attn_mask, label_embeded, label_enc_attn_mask)
            label_enc_attns.append(label_enc_attn)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns, label_enc_attns



class Transformer(nn.Module):
    def __init__(self, input_ids, atttention_mask, token_type_ids, label_embeded, label_input):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.input_ids = input_ids.to(device)
        self.atttention_mask = atttention_mask.to(device)
        self.token_type_ids = token_type_ids.to(device)
        self.label_embeded = label_embeded.to(device)
        self.label_input = label_input.to(device)
        self.fc = nn.Linear(768, 2).to(device)

    def forward(self):

        enc_outputs, enc_self_attns, label_enc_attns = self.encoder(self.input_ids, self.atttention_mask, self.token_type_ids, self.label_embeded, self.label_input)

        # enc_outputs = enc_outputs.mean(dim=1)
        # enc_outputs = self.fc(enc_outputs)
        return enc_outputs, enc_self_attns, label_enc_attns





if __name__ == '__main__':
    # pass

    test = ExtractAttributeFeature()

    for input_ids, atten_mask, image_feature, token_type_ids, label, group, id in LoadData.train_loader:
        input_ids, atten_mask, image_feature, token_type_ids, label = input_ids.to(device), atten_mask.to(device), \
                                                                      image_feature.to(device), token_type_ids.to(
            device), label

        label_embeded, label_input = test(label)  # [batch_size, seq_length, hidden_size]
        model = Transformer(input_ids, atten_mask, token_type_ids, label_embeded, label_input)
        enc_outputs, enc_self_attns, label_enc_attns = model()
        print(enc_outputs.shape)
#         loss_fn = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=1e-5)
#
#         optimizer.zero_grad()
#         print(group.shape)
#         loss = loss_fn(enc_outputs, group)
#         print("损失：", loss.item())
#         # print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
#         loss.backward(retain_graph=True)
#         optimizer.step()
#
#
#
#