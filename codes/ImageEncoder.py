import torch
from torch import nn, einsum
import torch.nn.functional as F
import LoadData
from TextEncoder import Transformer
from ImageFeature import ExtractImageFeature
from AttributeFeature import ExtractAttributeFeature
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0)  # batch_size x len_k, one is masking
    return pad_attn_mask  # batch_size x len_q


# tuple
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNormA(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm(q), self.norm(k), self.norm(v),**kwargs)


class PreNormFF(nn.Module):
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


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=4, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        x = lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h)  # q, k, v   (b, h, n, dim_head(64))
        q = x(q)
        k = x(k)
        v = x(v)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormA(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNormFF(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, q, k, v):

        for attn, ff in self.layers:
            qkv = attn(q, k, v) + q + k + v
            qkv = ff(qkv) + q + k + v
        return qkv


class ViT(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        num_patches = 1024
        assert pool in {'cls', 'mean'}
        self.heads = heads

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(1024, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))					# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim)
        )

        self.fc1 = nn.Linear(768, 1)
        self.fc2 = nn.Linear(1, 768)

    def forward(self, img1, img2, input_ids):
        attn_mask = get_attn_pad_mask(input_ids, input_ids)
        # print(attn_mask)
        # print(attn_mask.shape)
        x1 = self.to_patch_embedding(img1)
        x2 = img2
        x2 = self.fc1(x2).squeeze(2)
        x2.masked_fill_(attn_mask, -1e9)
        x2 = self.fc2(x2.unsqueeze(2))
        # print(x2.shape)  # [32, 196, 768]
        b, n, _ = x1.shape           # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x1 = torch.cat((cls_tokens, x1), dim=1)               # 将cls_token拼接到patch token中去
        x2 = torch.cat((cls_tokens, x2), dim=1)               # 将cls_token拼接到patch token中去
        x1 += self.pos_embedding[:, :(n+1)]                  # 加位置嵌入（直接加）
        x2 += self.pos_embedding[:, :(n+1)]                  # 加位置嵌入（直接加）
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)


        x = self.transformer(x2, x1, x1)
        # print(x.shape)

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # print(x.shape)

        return self.mlp_head(x)


#
# if __name__ == '__main__':
#     model_vit = ViT(
#         num_classes=768,
#         dim=768,
#         depth=6,
#         heads=16,
#         mlp_dim=2048,
#         dropout=0.1,
#         emb_dropout=0.1
#     )
#     """
#     patch embedding: 224*224 / 16*16 = 196
#     """
#
#
#     img1 = torch.randn(32, 196, 1024)
#     img2 = torch.randn(32, 196, 768)
#     # atten_mask = torch.randn(32, 196)
#
#
#     for input_ids, atten_mask, image_feature, token_type_ids, label, group, id in LoadData.train_loader:
#         preds  = model_vit(img1, img2, input_ids)
#         break
#
#
#     print(preds.shape)  # (32, 197, 768)

