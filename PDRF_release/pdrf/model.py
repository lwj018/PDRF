"""PDRF - Physically-Guided Data-space Rectified Flow (denoiser).

Data-space parameterization (MASD, paper Sec. 3.1): given the noisy state
Z_t, the flow time t and the historical context C, the network predicts
the clean future sequence  X1hat = f_theta(Z_t, t, C).
The CRF velocity is induced analytically: v = (X1hat - Z_t) / tbar,
which anchors the ODE trajectory to the valid data manifold.
"""

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from .blocks import (DownSample, UpSample, ResBlock, TimestepEmbedder,
                    OverlapPatchEmbed, D_SingleConv, Swish, swish)
from .kan_rwkv import shiftedBlock, VRWKV_Bottleneck
from .condition_encoder import ConditionEncoder
from .cgstf import ConditionGuidedSpatialTransformFusion
from .wgsc import WaveletGuidedSkipConnection


class PDRF(nn.Module):

    def __init__(self, T,in_ch,out_ch, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index h of bound'
        tdim = ch * 4
        self.time_embedding = TimestepEmbedder(tdim)
        attn = []
        self.head = nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record hput channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            h_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, h_ch=h_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            h_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, h_ch=h_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = h_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, out_ch, 3, stride=1, padding=1)
        )

        # 
        embed_dims = [128,256, 320]
        norm_layer = nn.LayerNorm
        dpr = [0.0, 0.0, 0.0]
        self.patch_embed3 = OverlapPatchEmbed(img_size=64 // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=64 // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])
        self.dnorm3 = norm_layer(embed_dims[1])

        self.kan_block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1],  mlp_ratio=1, drop_path=dpr[0], norm_layer=norm_layer),
            VRWKV_Bottleneck(
        n_embd=embed_dims[1],
        n_layer=8,
        layer_id=0,
        shift_mode='q_shift',
        channel_gamma=1/4,
        shift_pixel=1,
        hidden_rate=4,
        init_mode='fancy',
        drop_path=0.1,
        k_norm=True
    )])

        self.kan_block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2],  mlp_ratio=1, drop_path=dpr[1], norm_layer=norm_layer),
            VRWKV_Bottleneck(
        n_embd=embed_dims[2],
        n_layer=12,
        layer_id=0,
        shift_mode='q_shift',
        channel_gamma=1/4,
        shift_pixel=1,
        hidden_rate=4,
        init_mode='fancy',
        drop_path=0.1,
        k_norm=True
    )])

        self.kan_dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], mlp_ratio=1, drop_path=dpr[0], norm_layer=norm_layer),
            VRWKV_Bottleneck(
        n_embd=embed_dims[1],
        n_layer=8,
        layer_id=0,
        shift_mode='q_shift',
        channel_gamma=1/4,
        shift_pixel=1,
        hidden_rate=4,
        init_mode='fancy',
        drop_path=0.1,
        k_norm=True
    )])

        self.decoder1 = D_SingleConv(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_SingleConv(embed_dims[1], embed_dims[0])  


        self.encoder=ConditionEncoder()
        # self.encoder2=OFLMedSeg(input_channels=in_ch)
        # self.condition_fusion = ConditionFusion([64, 128, 128, 128, 256, 320])
      

        self.CGSTF0=ConditionGuidedSpatialTransformFusion(channels=64)
        self.CGSTF1=ConditionGuidedSpatialTransformFusion(channels=128)
        self.CGSTF2=ConditionGuidedSpatialTransformFusion(channels=128)
      


        #walete
        self.wgc3=WaveletGuidedSkipConnection(in_channels=128)
        self.wgc4=WaveletGuidedSkipConnection(in_channels=256)


        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t,condition,**kwargs):
        # Timestep embedding
        temb = self.time_embedding(t)

        #condition 
        enc=self.encoder(condition)
       
        # Downsampling
        h = self.head(x)
        hs = [h]

        i=0
        for layer in self.downblocks:
            h = layer(h, temb)
            if i in [1]:
                m = self.CGSTF0(h,enc[0])
                hs.append(m)
               
            elif i in [4]:
                m = self.CGSTF1(h,enc[1])
                hs.append(m)
            
            elif i in [7]:
                m = self.CGSTF2(h,enc[2])
                hs.append(m)

            else:
                hs.append(h)
            i=i+1
           

        #t3 = self.CGSTF3(h,enc[-3])
        t3=h
        

        B = x.shape[0]
        h, H, W = self.patch_embed3(h)
 
        for i, blk in enumerate(self.kan_block1):
            h = blk(h, H, W, temb)
            
        h = self.norm3(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #t4 = self.CGSTF4(h,enc[-2])
        t4=h
        

        h, H, W= self.patch_embed4(h)
       
        for i, blk in enumerate(self.kan_block2):
            h = blk(h, H, W, temb)
        h = self.norm4(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()    #[8, 320, 9, 9]
        h=h+enc[-1]



        ### Stage 4
        h = swish(F.interpolate(self.decoder1(h, temb), scale_factor=(2,2), mode ='bilinear'))

        h = self.wgc4(t4,h,enc[-2])

        _, _, H, W = h.shape
        h = h.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.kan_dblock1):
            h = blk(h, H, W, temb)
            

            
        ### Stage 3
        h = self.dnorm3(h)
        h = h.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()   #[8, 256, 18, 18]



        h = swish(F.interpolate(self.decoder2(h, temb),scale_factor=(2,2),mode ='bilinear'))   #[8, 128, 36, 36]
        h = self.wgc3(t3,h,enc[-3])




        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)          #[8, 128, 72, 72] 3   

           
        h = self.tail(h)

        assert len(hs) == 0
        return h


def build_pdrf(frames_in=5, frames_out=20, ch=64, ch_mult=(1, 2, 2, 2),
               attn=(), num_res_blocks=2, dropout=0.1, T=1000):
    """Build the PDRF denoiser with the configuration used in the paper."""
    return PDRF(T=T, in_ch=frames_out, out_ch=frames_out, ch=ch,
                ch_mult=list(ch_mult), attn=list(attn),
                num_res_blocks=num_res_blocks, dropout=dropout)


__all__ = ["PDRF", "build_pdrf"]
