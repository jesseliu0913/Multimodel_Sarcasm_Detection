import torch
from TextEncoder import Transformer
from ImageEncoder import ViT
from ImageFeature import ExtractImageFeature
from AttributeFeature import ExtractAttributeFeature
import LoadData
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.utils.data as Data
import torch.optim as optim



class ButtomFusion(nn.Module):
    def __init__(self):
        super(ButtomFusion, self).__init__()

        # extract image and label feature
        self.test_label = ExtractAttributeFeature()
        self.test_image = ExtractImageFeature()
        self.model_vit = ViT(
            num_classes=768,
            dim=768,
            depth=2,
            heads=4,
            mlp_dim=1024,
            dropout=0.1,
            emb_dropout=0.1
        )

    def __textSwitchFeature(self, text):
        text = text.mean(dim=1)
        return text


    def forward(self, label, input_ids, atten_mask, token_type_ids, image_feature):
        # label
        self.label = label
        self.input_ids = input_ids
        self.atten_mask = atten_mask
        self.token_type_ids = token_type_ids
        self.image_feature = image_feature
        label_embeded, label_input = self.test_label(self.label)  # label_embeded: [batch_size, seq_length, hidden_size]

        # text
        model_text = Transformer(self.input_ids, self.atten_mask, self.token_type_ids, label_embeded, label_input)
        enc_outputs, enc_self_attns, label_enc_attns = model_text()
        fst_fusion = enc_outputs # [batch_size, embedding_size)
        # print(enc_outputs.shape)  # [batch_size, seq_length, embedding_size]  32*196*768

        result, seq = self.test_image(self.image_feature)
        final_fusion = self.model_vit(seq, enc_outputs, self.input_ids) # 32*197*768
        # print(seq.shape) # [batch_size, ,embedding_size]  32*196*1024

        return fst_fusion, final_fusion


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(768, 2)

    def forward(self, hidden):
        output = self.fc(hidden) # [batch_size, 2]
        return output

# model = MyModel()
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=1e-5)
# # text_feature = []
# text_embedding = []
#
#
#
#
# for input_ids, atten_mask, image_feature, token_type_ids, label, group, id in LoadData.train_loader:
#     # print(atten_mask.shape)
#     buttom_fusion = ButtomFusion()
#     fst_fusion, final_fusion = buttom_fusion(label, input_ids, atten_mask, token_type_ids, image_feature)

#     loss = loss_fn(final_fusion, group)
#
#     print('损失:', loss.item())
#
#     loss.backward(retain_graph=True)
#     optimizer.step()
#     optimizer.zero_grad()


