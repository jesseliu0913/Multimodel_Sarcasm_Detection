import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from transformers import BertModel, BertConfig, BertTokenizer
import LoadData
import Config
from transformers import logging
logging.set_verbosity_error()


batch_size = Config.batch_size
device = Config.device
tokenizer = Config.tokenizer
model = Config.model
WORKING_PATH = Config.WORKING_PATH

class ExtractAttributeFeature(torch.nn.Module):
    def __init__(self):
        super(ExtractAttributeFeature, self).__init__()
        self.bert = model.to(device)
        # self.fc1 = nn.Linear(768, 128).to(device)

    def __getLabel(self, label):
        label_input_ids = []
        label_last_hidden_state = []

        for i in range(batch_size):
            label_input_list = []
            for j in label:
                label_input_list.append(j[i])
            label_input_str = list(map(str, label_input_list))
            label_input_str = ' '.join(label_input_str)
            last_hidden_state, input_ids = self.__getEmbedding(label_input_str)

            label_input_ids.append(input_ids.detach().tolist())
            label_last_hidden_state.append(last_hidden_state.detach().tolist())

        label_input_ids = torch.Tensor(label_input_ids).squeeze(1)
        label_last_hidden_state = torch.Tensor(label_last_hidden_state).squeeze(1)

        # print(label_input_ids.shape)  # [batch_size, seq_length]
        # print(label_last_hidden_state.shape)  # [batch_size, seq_length, hidden_size]
        return label_input_ids, label_last_hidden_state



    def __getEmbedding(self, x):
        result = tokenizer(x, padding='max_length', truncation=True, max_length=196, return_tensors='pt')
        # print(result)
        label_input = result["input_ids"]
        # print(laebl_atten_mask.shape)
        # print(result)
        input_ids, token_type_ids, attention_mask = result['input_ids'].to(device), result['token_type_ids'].to(device), result['attention_mask'].to(device)
        result = self.bert(input_ids, token_type_ids, attention_mask)
        return result.last_hidden_state, label_input


    def forward(self, label):
        label_input, word_embed = self.__getLabel(label)
        # label_input, word_embed = label_input.to(device), word_embed.to(device)
        # print(word_embed)
        # print(label_input.shape)
        # word_embed = self.fc1(word_embed)

        return word_embed, label_input



if __name__ == "__main__":
    test = ExtractAttributeFeature()
    for input_ids, atten_mask, image_feature, token_type_ids, label, group, id in LoadData.train_loader:
        input_ids, atten_mask, image_feature, token_type_ids, label = input_ids.to(device), atten_mask.to(device), \
                                                                      image_feature.to(device), token_type_ids.to(
            device), label
        label_embeded, label_input = test(label)
        print(label_embeded.shape)  # [batch_size, seq_length, hidden_size]

        break


# import torch
# import numpy as np
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import os
# from transformers import BertModel, BertConfig, BertTokenizer
# import LoadData
# import Config
# from transformers import logging
# logging.set_verbosity_error()
#
#
# batch_size = Config.batch_size
# device = Config.device
# tokenizer = Config.tokenizer
# model = Config.model
# WORKING_PATH = Config.WORKING_PATH
#
# class ExtractAttributeFeature(torch.nn.Module):
#     def __init__(self):
#         super(ExtractAttributeFeature, self).__init__()
#         self.bert = model
#         self.fc1 = nn.Linear(768, 128)
#
#     def __getLabel(self, label):
#         label_input_ids = []
#         label_last_hidden_state = []
#
#         for i in range(batch_size):
#             label_input_list = []
#             for j in label:
#                 label_input_list.append(j[i])
#             label_input_str = list(map(str, label_input_list))
#             label_input_str = ' '.join(label_input_str)
#             last_hidden_state, input_ids = self.__getEmbedding(label_input_str)
#
#             label_input_ids.append(input_ids.detach().tolist())
#             label_last_hidden_state.append(last_hidden_state.detach().tolist())
#
#         label_input_ids = torch.Tensor(label_input_ids).squeeze(1)
#         label_last_hidden_state = torch.Tensor(label_last_hidden_state).squeeze(1)
#
#         # print(label_input_ids.shape)  # [batch_size, seq_length]
#         # print(label_last_hidden_state.shape)  # [batch_size, seq_length, hidden_size]
#         return label_input_ids, label_last_hidden_state
#
#
#
#     def __getEmbedding(self, x):
#         result = tokenizer(x, padding='max_length', truncation=True, max_length=196, return_tensors='pt')
#         # print(result)
#         label_input = result["input_ids"]
#         # print(laebl_atten_mask.shape)
#         # print(result)
#         input_ids, token_type_ids, attention_mask = result['input_ids'], result['token_type_ids'], result['attention_mask']
#         result = self.bert(input_ids, token_type_ids, attention_mask)
#         return result.last_hidden_state, label_input
#
#
#     def forward(self, label):
#         label_input, word_embed = self.__getLabel(label)
#         word_embed = self.fc1(word_embed)
#         return word_embed, label_input
#
#
#
# if __name__ == "__main__":
#     test = ExtractAttributeFeature()
#     for input_ids, atten_mask, image_feature, token_type_ids, label, group, id in LoadData.train_loader:
#         # for i in label:
#         #     print(i[0])
#         label_embeded, label_input = test(label)
#         # print(label_embeded)  # [batch_size, seq_length, hidden_size]
#
#         break
#
#
