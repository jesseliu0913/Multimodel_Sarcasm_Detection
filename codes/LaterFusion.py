import torch
import torch.nn as nn
# from CoAttention import ParallelCoAttentionNetwork
from LWF import LWF
from AttributeFeature import ExtractAttributeFeature
import LoadData
from ButtomFusion import ButtomFusion
import torch.optim as optim
import Config
from torch.nn.parameter import Parameter


batch_size = Config.batch_size
device = Config.device

class LaterFusion(nn.Module):
    def __init__(self):
        super(LaterFusion, self).__init__()

        self.label_future = ExtractAttributeFeature()
        # self.co_atten = ParallelCoAttentionNetwork(768, 32)
        self.button_fusion = ButtomFusion()
        self.lwf = LWF()


    def __dealImage(self, img):
        img = torch.transpose(img, 1, 2).contiguous()
        return img

    def __dealPara(self, x):
        x = x.mean(dim=1)
        return x

    # def __coAttention(self, img, text):
    #     a_v, a_q, v, q = self.co_atten(img, text)
    #     return v, q

    def __Fusion(self, img, text, label):
        n = img.shape[0]
        # print(n)
        # print(img.shape)
        A = torch.ones(n, 1).to(device)
        img, text, label = img.to(device), text.to(device), label.to(device)

        img = torch.cat([img, A], dim=1).to(device)
        text = torch.cat([text, A], dim=1).to(device)
        label = torch.cat([label, A], dim=1).to(device)

        img = img.unsqueeze(2)
        text = label.unsqueeze(1)

        fusion_img_text = torch.einsum('nxt, nty->nxy', img, text)
        fusion_img_text = fusion_img_text.flatten(start_dim=1).unsqueeze(1)

        label = label.unsqueeze(1)
        fusion_all = torch.einsum('ntx, nty->nxy', fusion_img_text, label)
        fusion_all = fusion_all.flatten(start_dim=1)
        return fusion_all

    def __LWFusion(self, img, text, label):
        img, text, label = img.to(device), text.to(device), label.to(device)
        n = img.shape[0]
        A = torch.ones(n, 1).to(device)

        img = torch.cat([img, A], dim=1)
        text = torch.cat([text, A], dim=1)
        label = torch.cat([label, A], dim=1)

        # 假设所设秩: R = 4, 期望融合后的特征维度: h = 128
        R, h = 4, 128
        Wa = Parameter(torch.Tensor(R, img.shape[1], h).to(device))
        Wb = Parameter(torch.Tensor(R, text.shape[1], h).to(device))
        Wc = Parameter(torch.Tensor(R, label.shape[1], h).to(device))
        Wf = Parameter(torch.Tensor(1, R).to(device))
        bias = Parameter(torch.Tensor(1, h).to(device))

        # 分解后，并行提取各模态特征
        fusion_img = torch.matmul(img, Wa).to(device)
        fusion_text = torch.matmul(text, Wb).to(device)
        fusion_label = torch.matmul(label, Wc).to(device)

        # 利用一个Linear再进行特征融合（融合R维度）
        funsion_all = fusion_img * fusion_text * fusion_label
        funsion_all = torch.matmul(Wf, funsion_all.permute(1, 0, 2)).squeeze().to(device) + bias
        # print("11",funsion_all.shape)
        funsion_all = torch.where(torch.isnan(funsion_all), torch.full_like(funsion_all, 0), funsion_all)

        return funsion_all


    def forward(self, label, input_ids, atten_mask, token_type_ids, image_feature):
        self.label = label
        self.input_ids = input_ids
        self.atten_mask = atten_mask
        self.token_type_ids = token_type_ids
        self.image_feature = image_feature
        # text, img
        fst_fusion, final_fusion = self.button_fusion(self.label, self.input_ids, self.atten_mask, self.token_type_ids, self.image_feature)
        # fst_fusion = torch.randn(32, 196, 768)
        # final_fusion = torch.randn(32, 197, 768)
        img_future = self.__dealPara(final_fusion)
        # print(img_future.shape)
        # img_future = self.__dealImage(img_future)
        # print(img_future.shape)
        text_future = self.__dealPara(fst_fusion)
        # print(text_future.shape)
        #

        # label
        label_embeded, label_input = self.label_future(self.label)
        label_future = self.__dealPara(label_embeded)

        # coat_img, coat_text = self.__coAttention(img_future, text_future)
        # print(coat_img.shape)
        # print(coat_text.shape)

        # all_future = self.__Fusion(img_future, text_future, label_future)
        all_future = self.lwf(img_future, text_future, label_future)
        future_dim = all_future.size()[1]
        # print(future_dim)

        return all_future, future_dim


if __name__ == '__main__':
    model = LaterFusion().to(device)
    for input_ids, atten_mask, image_feature, token_type_ids, label, group, id in LoadData.train_loader:
        input_ids, atten_mask, image_feature, token_type_ids, label = input_ids.to(device), atten_mask.to(device), \
                                                                      image_feature.to(device), token_type_ids.to(
            device), label

        # print(label)
        # print(input_ids.shape)
        # print(atten_mask.shape)
        # print(token_type_ids.shape)
        # print(image_feature.shape)

        pred = model(label, input_ids, atten_mask, token_type_ids, image_feature)
        break



