import torch
import torch.nn as nn
from Models.resnet import ResNet
from decoder import decoder
from Multilevel_Gated_Interaction import Multilevel_Gated_Interaction

class GSHNet(nn.Module):
    def __init__(self,embed_dim=384,dim=96,img_size=224,method='model'):
        super(GSHNet, self).__init__()
        self.img_size = img_size
        self.feature_dims = []
        self.method = method
        self.dim = dim
        if method == 'model':
            self.encoder = ResNet()
            self.proj1 = nn.Conv2d(256,dim,1)
            self.proj2 = nn.Conv2d(512,dim*2,1)
            self.proj3 = nn.Conv2d(1024,dim*4,1)
            self.proj4 = nn.Conv2d(2048,dim*8,1)
            self.interact1 = Multilevel_Gated_Interaction(dim=dim*4,dim1=dim*8,embed_dim=embed_dim,num_heads=4,mlp_ratio=3)
            self.interact2 = Multilevel_Gated_Interaction(dim=dim*2,dim1=dim*4,dim2=dim*8,embed_dim=embed_dim,num_heads=2,mlp_ratio=3)
            self.interact3 = Multilevel_Gated_Interaction(dim=dim,dim1=dim*2,dim2=dim*4,embed_dim=embed_dim,num_heads=1,mlp_ratio=3)
            feature_dims=[dim,dim*2,dim*4]
        self.decoder = decoder(embed_dim=embed_dim,dims=feature_dims,img_size=img_size,mlp_ratio=1)

    def forward(self,x):
        fea = self.encoder(x)
        if self.method == 'model':
            fea_1_4,fea_1_8,fea_1_16,fea_1_32 = fea
            B,_,_,_ = fea_1_4.shape
            fea_1_4 = self.proj1(fea_1_4).reshape(B,self.dim,-1).transpose(1,2)
            fea_1_8 = self.proj2(fea_1_8).reshape(B,self.dim*2,-1).transpose(1,2)
            fea_1_16 = self.proj3(fea_1_16).reshape(B,self.dim*4,-1).transpose(1,2)
            fea_1_32 = self.proj4(fea_1_32).reshape(B,self.dim*8,-1).transpose(1,2)
            fea_1_16_ = self.interact1(fea_1_16,fea_1_32)
            fea_1_8_ = self.interact2(fea_1_8,fea_1_16_,fea_1_32)
            fea_1_4_ = self.interact3(fea_1_4,fea_1_8_,fea_1_16_)
        mask = self.decoder([fea_1_16_,fea_1_8_,fea_1_4_])
        return mask

    def flops(self):
        flops = 0
        flops += self.encoder.flops()
        N1 = self.img_size//4*self.img_size//4
        N2 = self.img_size//8*self.img_size//8
        N3 = self.img_size//16*self.img_size//16
        N4 = self.img_size//32*self.img_size//32
        flops += self.interact1.flops(N3,N4)
        flops += self.interact2.flops(N2,N3,N4)
        flops += self.interact3.flops(N1,N2,N3)
        flops += self.decoder.flops()
        return flops

if __name__ == '__main__':
    # Test
    model = GSHNet(embed_dim=384,dim=64,img_size=352,method='model')
    model.cuda()
