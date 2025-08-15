import torch
import torch.nn as nn
from timm.models.layers import DropPath
from Models.modules import GCA
import torch.nn.functional as F


def get_pos_embed(seq_len, dim):
    return nn.Parameter(torch.zeros(1, seq_len, dim))

class ASPP(nn.Module):
    def __init__(self, dim):
        super(ASPP, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, dilation=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Multilevel_Gated_Interaction(nn.Module):
    def __init__(self, dim, dim1, dim2=None, embed_dim=384, num_heads=6, mlp_ratio=3.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Multilevel_Gated_Interaction, self).__init__()
        self.dim = dim
        self.dim2 = dim2
        self.mlp_ratio = mlp_ratio

        self.interact1 = GCA(dim1=dim, dim2=dim1, dim=embed_dim,
                                        num_heads=num_heads, qkv_bias=qkv_bias,
                                        qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm0 = norm_layer(dim)
        self.norm1 = norm_layer(dim1)

        if self.dim2:
            self.interact2 = GCA(dim1=dim, dim2=dim2, dim=embed_dim,
                                            num_heads=num_heads, qkv_bias=qkv_bias,
                                            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.norm2 = norm_layer(dim2)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            act_layer(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        self.gate1 = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        if self.dim2:
            self.gate2 = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

        self.pos_embed = get_pos_embed(2304, dim)

        self.aspp = ASPP(dim)


    def forward(self, fea, fea_1, fea_2=None):
        B, N, _ = fea.shape

        if N != self.pos_embed.shape[1]:
            pos = F.interpolate(self.pos_embed.transpose(1, 2), size=N, mode='linear', align_corners=False).transpose(1, 2)
        else:
            pos = self.pos_embed

        pos = pos[:, :N, :]
        fea = self.norm0(fea + pos)

        fea_1 = self.norm1(fea_1)
        out1 = self.interact1(fea, fea_1)
        g1 = self.gate1(fea)
        fea = fea + g1 * out1

        if self.dim2:
            fea_2 = self.norm2(fea_2)
            out2 = self.interact2(fea, fea_2)
            g2 = self.gate2(fea)
            fea = fea + g2 * out2
        fea = fea + self.drop_path(self.mlp(self.norm(fea)))
        fea = self.aspp(fea)
        return fea


if __name__ == '__main__':
    # Test
    model = Multilevel_Gated_Interaction(dim1=96,dim2=192,dim3=384)
    model.cuda()


