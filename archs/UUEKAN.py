import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st

from .kan import KANLinear, KAN
from torch.nn import init

# Import uncertainty module
from .UncertainModule.uncertainty_module import (
    UncertaintyGuidedProcessor, 
    UncertaintyRefinementAttention,
    create_uncertainty_processor,
    create_uncertainty_attention
)


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                        in_features,
                        hidden_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc2 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )
            self.fc3 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )   
            self.fc4 = KANLinear(
                        hidden_features,
                        out_features,
                        grid_size=grid_size,
                        spline_order=spline_order,
                        scale_noise=scale_noise,
                        scale_base=scale_base,
                        scale_spline=scale_spline,
                        base_activation=base_activation,
                        grid_eps=grid_eps,
                        grid_range=grid_range,
                    )   

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        # TODO
        # self.fc1 = nn.Linear(in_features, hidden_features)


        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
        self.dwconv_4 = DW_bn_relu(hidden_features)
    
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_3(x, H, W)
        x = self.fc4(x.reshape(B*N,C))
        x = x.reshape(B,N,C).contiguous()
        x = self.dwconv_4(x, H, W)
    
        return x

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x


class FeatureEdgeExtractor(nn.Module):
    """
    Feature-level edge extractor
    Used to extract edge features after the KAN layer output
    """
    def __init__(self, dim):
        super().__init__()
        
        # Sobel operator for edge detection
        self.sobel_x = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False, groups=dim)
        self.sobel_y = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False, groups=dim)
        
        # Initialize Sobel convolution kernels
        sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        for i in range(dim):
            self.sobel_x.weight.data[i, 0] = sobel_kernel_x
            self.sobel_y.weight.data[i, 0] = sobel_kernel_y
        
        # Freeze Sobel weights
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False
        
        # Edge feature fusion layer (learnable)
        self.edge_fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        # Edge attention
        self.edge_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.weight.requires_grad:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x, H, W):
        """
        Args:
            x: Feature in [B, N, C] format (from KAN layer output)
            H, W: Height and width of the feature map
        Returns:
            edge_features: Edge-enhanced features in [B, N, C] format
        """
        B, N, C = x.shape
        
        # Convert to [B, C, H, W] format
        x_spatial = x.transpose(1, 2).view(B, C, H, W)
        
        # Sobel edge detection
        edge_x = self.sobel_x(x_spatial)
        edge_y = self.sobel_y(x_spatial)
        
        # Fuse edges
        edge_combined = torch.cat([edge_x, edge_y], dim=1)
        edge_features = self.edge_fusion(edge_combined)
        
        # Attention weighting
        attention = self.edge_attention(edge_features)
        edge_features = edge_features * attention
        
        # Convert back to [B, N, C] format
        edge_features = edge_features.flatten(2).transpose(1, 2)
        
        return edge_features


class EKANBlock(nn.Module):
    """
    Edge-KAN Block: First extract edge features, then fuse and input to KAN.
    Process: Input -> Edge Extraction -> Feature Fusion -> KAN Processing -> Output
    """
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
                 no_kan=False, use_edge=True, edge_weight=0.5):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)  # Normalization before edge extraction
        self.norm2 = norm_layer(dim)  # Normalization before KAN input
        mlp_hidden_dim = int(dim)
        
        self.use_edge = use_edge
        self.edge_weight = edge_weight

        # Edge extractor (extract first)
        if self.use_edge:
            self.edge_extractor = FeatureEdgeExtractor(dim)
        
        # KAN layer (receives fused features)
        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, 
                             act_layer=act_layer, drop=drop, no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            if m.weight.requires_grad:
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, H, W):
        """
        New forward propagation process (conservative version - using add for fusion):
        1. Extract edge features from the original input.
        2. Add original features and edge features for fusion.
        3. Fused features are sent to KAN for processing.
        4. Residual connection.
        """
        # Normalize input
        x_norm = self.norm1(x)
        
        if self.use_edge:
            # Step 1: Extract edge information from original features
            edge_features = self.edge_extractor(x_norm, H, W)
            
            # Step 2: Feature fusion - use add method (more conservative)
            # Weighted sum, does not change feature dimension
            fused_features = x_norm + self.edge_weight * edge_features
            
            # Step 3: Fused features sent to KAN
            kan_out = self.layer(self.norm2(fused_features), H, W)
        else:
            # If not using edges, pass directly through KAN
            kan_out = self.layer(self.norm2(x_norm), H, W)
        
        # Step 4: Residual connection
        x = x + self.drop_path(kan_out)
        
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)



class UUEKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], use_uncertainty=True, **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]
        self.use_uncertainty = use_uncertainty
        self.img_size = img_size

        self.encoder1 = ConvLayer(3, kan_input_dim//8)  
        self.encoder2 = ConvLayer(kan_input_dim//8, kan_input_dim//4)  
        self.encoder3 = ConvLayer(kan_input_dim//4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Use EKANBlock (Edge-KAN Block) instead of the original KANBlock
        # EKANBlock adds edge extraction after the KAN layer, output = KAN output + edge_weight * edge features
        self.block1 = nn.ModuleList([EKANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, no_kan=no_kan,
            use_edge=True, edge_weight=0.5
            )])

        self.block2 = nn.ModuleList([EKANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer, no_kan=no_kan,
            use_edge=True, edge_weight=0.5
            )])

        self.dblock1 = nn.ModuleList([EKANBlock(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, no_kan=no_kan,
            use_edge=True, edge_weight=0.5
            )])

        self.dblock2 = nn.ModuleList([EKANBlock(
            dim=embed_dims[0], 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer, no_kan=no_kan,
            use_edge=True, edge_weight=0.5
            )])

        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])  
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0]//4) 
        self.decoder4 = D_ConvLayer(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer(embed_dims[0]//8, embed_dims[0]//8)

        self.final = nn.Conv2d(embed_dims[0]//8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim =1)
        
        # Uncertainty module
        if self.use_uncertainty:
            # Uncertainty processor
            self.uncertainty_processor = create_uncertainty_processor(
                base_size=(img_size, img_size), 
                uncertainty_threshold=0.5
            )
            
            # Create uncertainty attention module for each skip connection layer
            self.uncertainty_attention_4 = UncertaintyRefinementAttention(
                in_channel=embed_dims[1], 
                out_channel=1, 
                dim=embed_dims[1], 
                base_size=(img_size//8, img_size//8),
                stage=4
            )
            
            self.uncertainty_attention_3 = UncertaintyRefinementAttention(
                in_channel=embed_dims[0], 
                out_channel=1, 
                dim=embed_dims[0], 
                base_size=(img_size//4, img_size//4),
                stage=3
            )
            
            self.uncertainty_attention_2 = UncertaintyRefinementAttention(
                in_channel=embed_dims[0]//4, 
                out_channel=1, 
                dim=embed_dims[0]//4, 
                base_size=(img_size//2, img_size//2),
                stage=2
            )
            
            self.uncertainty_attention_1 = UncertaintyRefinementAttention(
                in_channel=embed_dims[0]//8, 
                out_channel=1, 
                dim=embed_dims[0]//8, 
                base_size=(img_size, img_size),
                stage=1
            )
            
            # Convolutional layers to generate uncertainty maps for each stage
            self.uncertainty_conv_4 = nn.Conv2d(embed_dims[1], 1, kernel_size=1)
            self.uncertainty_conv_3 = nn.Conv2d(embed_dims[0], 1, kernel_size=1)  
            self.uncertainty_conv_2 = nn.Conv2d(embed_dims[0]//4, 1, kernel_size=1)
            self.uncertainty_conv_1 = nn.Conv2d(embed_dims[0]//8, 1, kernel_size=1)
            self.uncertainty_sigmoid = nn.Sigmoid()
    
    def to(self, device):
        """Override the to method to ensure the uncertainty module is also moved to the correct device"""
        super().to(device)
        if hasattr(self, 'use_uncertainty') and self.use_uncertainty:
            if hasattr(self, 'uncertainty_processor'):
                self.uncertainty_processor.to(device)
        return self
        
    def cuda(self, device=None):
        """Override the cuda method to ensure the uncertainty module is also moved to CUDA"""
        super().cuda(device)
        if hasattr(self, 'use_uncertainty') and self.use_uncertainty:
            if hasattr(self, 'uncertainty_processor'):
                if device is None:
                    self.uncertainty_processor.cuda()
                else:
                    self.uncertainty_processor.cuda(device)
        return self

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        ### Tokenized KAN Stage
        ### Stage 4

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out, H, W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Decoder with Uncertainty-guided Skip Connections
        
        ### Stage 4 - First decoder layer with uncertainty-guided skip connection
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2), mode ='bilinear'))
        
        if self.use_uncertainty:
            # Generate initial uncertainty map
            saliency_map = self.uncertainty_conv_4(out)
            saliency_map = self.uncertainty_sigmoid(saliency_map)
            
            # Ensure the uncertainty processor is on the correct device
            if not hasattr(self.uncertainty_processor.uncertainty_generator, '_device_synced'):
                self.uncertainty_processor.to(out.device)
                self.uncertainty_processor.uncertainty_generator._device_synced = True
            
            # Generate uncertainty map
            uncertainty_map_4 = self.uncertainty_processor.process(saliency_map, (out.shape[2], out.shape[3]))
            
            # Apply uncertainty-guided attention mechanism to the skip connection
            refined_t4, _, partition_map_4, _, _ = self.uncertainty_attention_4(
                t4, out, uncertainty_map_4
            )
            out = torch.add(out, refined_t4)
        else:
            out = torch.add(out, t4)
            
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3 - Second decoder layer with uncertainty-guided skip connection
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        
        if self.use_uncertainty:
            # Update uncertainty map
            saliency_map = self.uncertainty_conv_3(out)
            saliency_map = self.uncertainty_sigmoid(saliency_map)
            uncertainty_map_3 = self.uncertainty_processor.process(saliency_map, (out.shape[2], out.shape[3]))
            
            # Apply uncertainty-guided attention mechanism to the skip connection
            refined_t3, _, partition_map_3, _, _ = self.uncertainty_attention_3(
                t3, out, uncertainty_map_3
            )
            out = torch.add(out, refined_t3)
        else:
            out = torch.add(out, t3)
            
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 2 - Third decoder layer with uncertainty-guided skip connection
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        
        if self.use_uncertainty:
            # Update uncertainty map
            saliency_map = self.uncertainty_conv_2(out)
            saliency_map = self.uncertainty_sigmoid(saliency_map)
            uncertainty_map_2 = self.uncertainty_processor.process(saliency_map, (out.shape[2], out.shape[3]))
            
            # Apply uncertainty-guided attention mechanism to the skip connection
            refined_t2, _, partition_map_2, _, _ = self.uncertainty_attention_2(
                t2, out, uncertainty_map_2
            )
            out = torch.add(out, refined_t2)
        else:
            out = torch.add(out, t2)
            
        ### Stage 1 - Fourth decoder layer with uncertainty-guided skip connection
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        
        if self.use_uncertainty:
            # Update uncertainty map
            saliency_map = self.uncertainty_conv_1(out)
            saliency_map = self.uncertainty_sigmoid(saliency_map)
            uncertainty_map_1 = self.uncertainty_processor.process(saliency_map, (out.shape[2], out.shape[3]))
            
            # Apply uncertainty-guided attention mechanism to the skip connection
            refined_t1, _, partition_map_1, _, _ = self.uncertainty_attention_1(
                t1, out, uncertainty_map_1
            )
            out = torch.add(out, refined_t1)
        else:
            out = torch.add(out, t1)
            
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)

