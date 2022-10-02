import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.init import xavier_normal_
import Config
# from torch.nn.parameter import Parameter

batch_size = Config.batch_size
device = Config.device



class LWF(nn.Module):
    def __init__(self):
        super(LWF, self).__init__()
        self.R = 4
        self.h = 128
        self.batch_size = batch_size



    def forward(self, A, B, C):
        A, B, C = A.to(device), B.to(device), C.to(device)
        An = A.shape[0]
        Bn = B.shape[0]
        Cn = C.shape[0]

        S_a = torch.ones(An, 1).to(device)
        S_b = torch.ones(Bn, 1).to(device)
        S_c = torch.ones(Cn, 1).to(device)

        A = torch.cat([A, S_a], dim=1)
        B = torch.cat([B, S_b], dim=1)
        C = torch.cat([C, S_c], dim=1)

        Wa = Parameter(torch.Tensor(self.R, A.shape[1], self.h)).to(device)
        Wb = Parameter(torch.Tensor(self.R, B.shape[1], self.h)).to(device)
        Wc = Parameter(torch.Tensor(self.R, C.shape[1], self.h)).to(device)
        Wf = Parameter(torch.Tensor(1, self.R)).to(device)
        bias = Parameter(torch.Tensor(1, self.h)).to(device)

        xavier_normal_(Wa).to(device)
        xavier_normal_(Wb).to(device)
        xavier_normal_(Wc).to(device)
        xavier_normal_(Wf).to(device)
        bias.data.fill_(0).to(device)

        fusion_A = torch.matmul(A, Wa).to(device)
        fusion_B = torch.matmul(B, Wb).to(device)
        fusion_C = torch.matmul(C, Wc).to(device)

        funsion_ABC = fusion_A * fusion_B * fusion_C
        funsion_ABC = torch.matmul(Wf, funsion_ABC.permute(1, 0, 2)).squeeze().to(device) + bias
        funsion_ABC = funsion_ABC.view(-1, self.h).to(device)
        # funsion_ABC = torch.where(torch.isnan(funsion_ABC), torch.full_like(funsion_ABC, 0), funsion_ABC)
        return funsion_ABC


# A = torch.randn(32, 512).to(device)
# # print(A)
# B = torch.randn(32, 1024).to(device)
# # print(B)
# C = torch.randn(32, 32).to(device)
# # print(C)
# # R, h = 4, 128
#
# lwf = LWF().to(device)
# fusion = lwf(A, B, C)
# print(fusion)




# A = torch.cat([A, torch.ones(n, 1)], dim=1)
# B = torch.cat([B, torch.ones(n, 1)], dim=1)
# C = torch.cat([C, torch.ones(n, 1)], dim=1)
# DTYPE = torch.cuda.FloatTensor




