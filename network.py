import timm
from timm.layers import resample_abs_pos_embed_nhwc
import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba
import torch.nn.functional as F

act_func = nn.GELU()
act_params = ("gelu")
class _LoRA_qkv_timm(nn.Module):
    def __init__(
        self,
        qkv: nn.Module,
        r: int
    ):
        super().__init__()
        self.r = r
        self.qkv = qkv
        self.dim = qkv.in_features
        self.linear_a_q = nn.Linear(self.dim, r, bias=False)
        self.linear_b_q = nn.Linear(r, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, r, bias=False)
        self.linear_b_v = nn.Linear(r, self.dim, bias=False)
        self.act = act_func
        self.w_identity = torch.eye(self.dim)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.linear_a_q.weight, a=math.sqrt(5))     
        nn.init.kaiming_uniform_(self.linear_a_v.weight, a=math.sqrt(5))  
        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.zeros_(self.linear_b_v.weight)
    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.act(self.linear_a_q(x)))
        new_v = self.linear_b_v(self.act(self.linear_a_v(x)))
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv
    
class Mamba_Layer(nn.Module):
    def __init__(self, token_dim=256, channel_dim=768, channel_reduce=3, token_reduce=8):
        super().__init__()
        self.token_dim = token_dim
        self.channel_dim = channel_dim
        self.channel_r_dim = channel_dim // channel_reduce
        self.token_r_dim = token_dim // token_reduce
        self.channel_downsample = nn.Linear(self.channel_dim, self.channel_r_dim, bias=False)
        self.norm1 = nn.LayerNorm(self.channel_r_dim)
        self.mamba1 = Mamba(
                d_model=self.channel_r_dim, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
        self.channel_upsample = nn.Linear(self.channel_r_dim, self.channel_dim, bias=False)

        self.token_downsample = nn.Linear(self.token_dim, self.token_r_dim, bias=False)
        self.norm2 = nn.LayerNorm(self.token_r_dim)
        self.mamba2 = Mamba(
                d_model=self.token_r_dim, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            )
        self.token_upsample = nn.Linear(self.token_r_dim, self.token_dim, bias=False)

    def forward(self, x):

        x1 = self.norm1(self.channel_downsample(x))
        x1 = self.channel_upsample(self.mamba1(x1)) + x

        x2 = self.norm2(self.token_downsample(x1.transpose(-1, -2)))
        x2 = self.token_upsample(self.mamba2(x2)).transpose(-1, -2) + x1

        return x2

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        y = self.weight[:, None, None] * x
        # y = torch.mul(self.weight[:, None, None], x)
        x = y + self.bias[:, None, None]
        return x



class SegModel(nn.Module):
    def __init__(self, 
    encoder_embed_dim: int = 1024,
    pretrain_model: str = 'samvit_large_patch16',
    out_chans: int = 1024,
    depth: int = 24,
    pretrained: bool = True,
    freeze_encoder: bool = True,
    input_size:int = 1024,
    deep_supervision: bool = False,
    ) -> None:

        super().__init__()
        self.encoder_embed_dim = encoder_embed_dim
        self.depth = depth
        self.pretrain_model = pretrain_model
        self.deep_supervision = deep_supervision
        self.sam_encoder = timm.create_model(self.pretrain_model, pretrained=pretrained, num_classes=0)
        
        if freeze_encoder:
            for name, param in self.sam_encoder.named_parameters():
                param.requires_grad = False

        for layer_i, blk in enumerate(self.sam_encoder.blocks):
            self.sam_encoder.blocks[layer_i].attn.qkv = _LoRA_qkv_timm(blk.attn.qkv, 128)

        token_dim = (input_size//16) * (input_size//16)
        if input_size >= 512 and input_size < 1024:
            token_reduce = 4
        if input_size==1024:
            token_reduce = 8
        if input_size < 512:
            token_reduce = 1

        self.mamba_layer = nn.ModuleList([Mamba_Layer(token_dim=token_dim, channel_dim=encoder_embed_dim, token_reduce=token_reduce) for i in range(4)]) 

        self.neck = nn.Sequential(
            nn.Conv2d(self.encoder_embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_embed_dim, out_chans//8, kernel_size=16, stride=8, padding=4, bias=False),
            LayerNorm2d(out_chans//8),
            nn.Conv2d(out_chans//8, out_chans//8, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans//8),
        )

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_embed_dim, out_chans//4, kernel_size=4, stride=4, padding=0, bias=False),
            LayerNorm2d(out_chans//4),
            nn.Conv2d(out_chans//4, out_chans//4, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans//4),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_embed_dim, out_chans//2, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNorm2d(out_chans//2),
            nn.Conv2d(out_chans//2, out_chans//2, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans//2),
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(out_chans, out_chans//2, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNorm2d(out_chans//2),
            nn.ReLU(),
            nn.Conv2d(out_chans//2, out_chans//2, kernel_size=1, padding=0, bias=False),
            LayerNorm2d(out_chans//2),
        )


        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(out_chans, out_chans//4, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNorm2d(out_chans//4),
            nn.ReLU(),
            nn.Conv2d(out_chans//4, out_chans//4, kernel_size=1, padding=0, bias=False),
            LayerNorm2d(out_chans//4),
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(out_chans//2, out_chans//8, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNorm2d(out_chans//8),
            nn.ReLU(),
            nn.Conv2d(out_chans//8, out_chans//8, kernel_size=1, padding=0, bias=False),
            LayerNorm2d(out_chans//8),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(out_chans//4, out_chans//16, kernel_size=4, stride=2, padding=1, bias=False),
            LayerNorm2d(out_chans//16),
            nn.ReLU(),
            nn.Conv2d(out_chans//16, out_chans//16, kernel_size=1, padding=0, bias=False),
            LayerNorm2d(out_chans//16),
        )

        if self.deep_supervision:
            self.out_conv1 = nn.Sequential(
                nn.Conv2d(out_chans//2, out_chans//2, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_chans//2),
                nn.ReLU(),
                nn.Conv2d(out_chans//2, 1, kernel_size=1, padding=0, bias=False),
            )

            self.out_conv2 = nn.Sequential(
                nn.Conv2d(out_chans//4, out_chans//4, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_chans//4),
                nn.ReLU(),
                nn.Conv2d(out_chans//4, 1, kernel_size=1, padding=0, bias=False),
            )

            self.out_conv3 = nn.Sequential(
                nn.Conv2d(out_chans//8, out_chans//8, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_chans//8),
                nn.ReLU(),
                nn.Conv2d(out_chans//8, 1, kernel_size=1, padding=0, bias=False),
            )

            self.out_conv4 = nn.Sequential(
                nn.Conv2d(out_chans//16, out_chans//16, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_chans//16),
                nn.ReLU(),
                nn.Conv2d(out_chans//16, 1, kernel_size=1, padding=0, bias=False),
            )

        else:
            self.final = nn.Sequential(
                nn.Conv2d(out_chans//16, out_chans//16, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(out_chans//16),
                nn.ReLU(),
                nn.Conv2d(out_chans//16, 1, kernel_size=1, padding=0, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_shape = x.shape
        x = self.sam_encoder.patch_embed(x)
        if self.sam_encoder.pos_embed is not None:
            x = x + resample_abs_pos_embed_nhwc(self.sam_encoder.pos_embed, x.shape[1:3])
        x = self.sam_encoder.pos_drop(x)
        x = self.sam_encoder.patch_drop(x)
        x = self.sam_encoder.norm_pre(x)

        pyramid_feat = []
        for i in range(self.depth):
            x = self.sam_encoder.blocks[i](x)
            if (i+1) % (self.depth//4) == 0:
                b,h,w,c = x.size()
                x = x.reshape(b, -1, c)
                x = self.mamba_layer[i//(self.depth//4)](x)
                x = x.reshape(b, h, w, c)
                pyramid_feat.append(x.permute(0, 3, 1, 2))

        up8 = self.up8(pyramid_feat[0])  #
        up4 = self.up4(pyramid_feat[0] + pyramid_feat[1])
        up2 = self.up2(pyramid_feat[0] + pyramid_feat[1] + pyramid_feat[2])
        x = self.neck(pyramid_feat[0] + pyramid_feat[1] + pyramid_feat[2] + pyramid_feat[-1])

        d1 = self.deconv1(x)
        d2 = self.deconv2(torch.cat([up2, d1], dim=1))
        d3 = self.deconv3(torch.cat([up4, d2], dim=1))
        d4 = self.deconv4(torch.cat([up8, d3], dim=1))
  
        if self.deep_supervision:
            up_out1 = F.interpolate(d1, size=img_shape[-2:], mode ='bilinear',align_corners=True)  # 512 32 32
            out1 = self.out_conv1(up_out1)

            up_out2 = F.interpolate(d2, size=img_shape[-2:], mode ='bilinear',align_corners=True)  # 256,64,64
            up_out2 = up_out2 * out1
            out2 = self.out_conv2(up_out2)

            up_out3 = F.interpolate(d3 ,size=img_shape[-2:], mode ='bilinear',align_corners=True) #128,128,128
            up_out3 = up_out3 * out2
            out3 = self.out_conv3(up_out3)

            up_out4 = F.interpolate(d4 ,size=img_shape[-2:], mode ='bilinear',align_corners=True) #64,256,256
            up_out4 = up_out4 * out3
            out4 = self.out_conv4(up_out4)

            return [out1, out2, out3, out4]
        
        else:
            return self.final(d4)



if __name__ == '__main__':
    x = torch.rand((4,3,256,256)).to('cuda')
    model = SegModel(input_size=256).to('cuda')
    print(model(x)[-1].shape)