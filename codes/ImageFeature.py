import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

import LoadData


class ExtractImageFeature(nn.Module):
    def __init__(self):
        super(ExtractImageFeature, self).__init__()
        # 2048->1024
        self.Linear = torch.nn.Linear(2048, 1024)

    def forward(self, input):
        input = input.permute(1, 0, 2)
        output = list()
        for i in range(196):
            sub_output = torch.nn.functional.relu(self.Linear(input[i]))
            output.append(sub_output)
        output = torch.stack(output)  # torch.stack()是对tensors沿指定维度拼接，但返回的Tensor会多一维
        # print(output.shape)
        mean = torch.mean(output, 0)
        return mean, output.permute(1, 0, 2)


# test = ExtractImageFeature()

# if __name__ == "__main__":
#     test = ExtractImageFeature()
#     for input_ids, atten_mask, image_feature, token_type_ids, label, group, id in LoadData.train_loader:
#         result, seq = test(image_feature)
#         # [1, 1024]
#         print(result.shape)
#         # [1, 196, 1024]
#         print(seq.shape)
#         break


