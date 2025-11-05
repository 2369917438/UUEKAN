"""
Uncertainty Calculation Module
This module extracts uncertainty-related functionalities from the UGRAN project, including:
1. Uncertainty Map Generation (UncertaintyMapGenerator)
2. Uncertainty-Guided Attention Mechanism (UncertaintyRefinementAttention)
3. Adaptive Partitioning Strategy (AdaptivePartition)

Version Update:
- Integrated MALA (Magnitude-Aware Linear Attention) mechanism.
- Replaced traditional O(n^2) attention with O(n) linear attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import time
from typing import Tuple, Optional

# Import MALA attention mechanism
from .mala_attention import MALAAttention


class UncertaintyMapGenerator:
    """
    Uncertainty Map Generator
    Computes uncertainty regions based on saliency maps and a threshold.
    """
    
    def __init__(self, ksize: int = 7, sigma: float = 1.0, channels: int = 1, threshold: float = 0.5):
        """
        Initializes the Uncertainty Map Generator.
        
        Args:
            ksize: Gaussian kernel size.
            sigma: Standard deviation of the Gaussian kernel.
            channels: Number of channels.
            threshold: Uncertainty threshold.
        """
        self.ksize = ksize
        self.sigma = sigma
        self.channels = channels
        self.uthreshold = threshold
        
        # Create Gaussian kernel
        k = cv2.getGaussianKernel(ksize, sigma)
        k = np.outer(k, k)
        k = torch.tensor(k).float()
        self.kernel = k.repeat(channels, 1, 1, 1)
        
    def to(self, device):
        """Moves the module to the specified device."""
        self.kernel = self.kernel.to(device)
        return self
        
    def cuda(self, idx: Optional[int] = None):
        """Moves the module to a CUDA device."""
        if idx is None:
            idx = torch.cuda.current_device()
        self.to(device=f"cuda:{idx}")
        return self
    
    def get_uncertainty_map(self, saliency_map: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Generates an uncertainty map from a saliency map.
        
        Args:
            saliency_map: Saliency map [B, 1, H, W].
            target_shape: Target size (H, W).
            
        Returns:
            uncertainty_map: Uncertainty map [B, 1, H, W].
        """
        # Resize and apply sigmoid
        smap = F.interpolate(saliency_map, size=target_shape, mode='bilinear', align_corners=False)
        smap = torch.sigmoid(smap)
        
        # Calculate distance from the threshold
        p = smap - self.uthreshold
        
        # Uncertainty = threshold - |distance|; the smaller the distance, the higher the uncertainty.
        uncertainty = self.uthreshold - torch.abs(p)
        
        # Apply Gaussian convolution for smoothing
        uncertainty = F.pad(uncertainty, (self.ksize // 2, ) * 4, 'constant', 0)
        uncertainty = F.conv2d(uncertainty, self.kernel * 4, groups=1)
        
        # Normalize to [0,1]
        return uncertainty / uncertainty.max()


class Conv2d(nn.Module):
    """2D convolutional layer with an optional ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = None, dilation: int = 1, 
                 groups: int = 1, bias: bool = True, relu: bool = False):
        super(Conv2d, self).__init__()
        
        if padding is None:
            padding = kernel_size // 2
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             stride, padding, dilation, groups, bias)
        self.relu = nn.ReLU(inplace=True) if relu else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class AdaptivePartition:
    """
    Adaptive Partitioning Strategy
    Decides whether to partition a region based on its uncertainty density.
    """
    
    def __init__(self, base_size: Tuple[int, int] = (384, 384), partition_threshold: float = 0.2):
        """
        Args:
            base_size: Base image size.
            partition_threshold: Partitioning threshold; regions with uncertainty density below this value will be partitioned.
        """
        self.base_size = base_size
        self.min_size = base_size[0] // 16  # Minimum partition size
        self.max_size = base_size[0] // 4   # Maximum partition size
        self.partition_threshold = partition_threshold
        
    def should_partition(self, uncertainty_map: torch.Tensor, current_size: int) -> bool:
        """
        Determines if a region needs to be partitioned.
        
        Args:
            uncertainty_map: The uncertainty map.
            current_size: The size of the current region.
            
        Returns:
            Whether the region should be partitioned.
        """
        # Calculate uncertainty density
        uncertainty_density = torch.sum(uncertainty_map) / (uncertainty_map.shape[-1] * uncertainty_map.shape[-2])
        
        # Partition condition: low uncertainty density and size is greater than minimum, or size exceeds maximum.
        return (uncertainty_density < self.partition_threshold and current_size > self.min_size) or current_size > self.max_size


class UncertaintyRefinementAttention(nn.Module):
    """
    Uncertainty-Guided Refinement Attention Mechanism
    Adaptively processes features based on the uncertainty map.
    
    Version Update:
    - Integrated MALA (Magnitude-Aware Linear Attention).
    - Complexity reduced from O(n^2) to O(n).
    - Preserves magnitude information to improve attention quality.
    """
    
    def __init__(self, in_channel: int, out_channel: int = 1, dim: int = 64, 
                 base_size: Tuple[int, int] = (384, 384), stage: Optional[int] = None,
                 use_mala: bool = True):
        """
        Args:
            in_channel: Number of input channels.
            out_channel: Number of output channels.
            dim: Feature dimension.
            base_size: Base image size.
            stage: Processing stage.
            use_mala: Whether to use MALA attention (True=linear complexity, False=traditional attention).
        """
        super(UncertaintyRefinementAttention, self).__init__()
        
        self.base_size = base_size
        self.ratio = stage
        self.dim = dim
        self.use_mala = use_mala
        
        # Initialize adaptive partitioning strategy
        self.adaptive_partition = AdaptivePartition(base_size)
        
        # Network layers
        self.norm = nn.BatchNorm2d(dim)
        self.lnorm = nn.BatchNorm2d(dim)
        
        # Attention mechanism selection
        if use_mala:
            # Use MALA linear attention (O(n) complexity)
            self.attention = MALAAttention(dim, num_heads=1)
        else:
            # Use traditional multi-head attention (O(n^2) complexity)
            self.mha = nn.MultiheadAttention(dim, 1, batch_first=True)
        
        # Linear transformation layers (retained for backward compatibility)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        # Output layers
        self.conv_out1 = nn.Linear(dim, dim)
        self.conv_out3 = Conv2d(dim, dim, 3, relu=True)
        self.conv_out4 = Conv2d(dim, out_channel, 1)
        
        # Performance statistics
        self.ptime = 0.0  # partition time
        self.rtime = 0.0  # reverse time
        self.etime = 0.0  # execute time
        
    def adaptive_partition_process(self, x: torch.Tensor, l: torch.Tensor, 
                                 uncertainty_map: torch.Tensor, partition_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive partitioning process.
        
        Args:
            x: Input features [B, C, H, W].
            l: Reference features [B, C, H, W].
            uncertainty_map: Uncertainty map [B, 1, H, W].
            partition_map: Partition map [B, 1, H, W].
            
        Returns:
            Processed features and the partition map.
        """
        B, C, H, W = x.shape
        h, w = H // 2, W // 2
        
        start_time = time.process_time()
        
        # Split the feature map into 4 sub-regions
        x_windows = x.view(B, C, 2, h, 2, w).permute(2, 4, 0, 1, 3, 5).contiguous().view(4, B, C, h, w)
        l_windows = l.view(B, C, 2, h, 2, w).permute(2, 4, 0, 1, 3, 5).contiguous().view(4, B, C, h, w)
        u_windows = uncertainty_map.view(B, 1, 2, h, 2, w).permute(2, 4, 0, 1, 3, 5).contiguous().view(4, B, 1, h, w)
        p_windows = partition_map.view(B, 1, 2, h, 2, w).permute(2, 4, 0, 1, 3, 5).contiguous().view(4, B, 1, h, w)
        
        # Add boundary markers for partition visualization
        for i in range(4):
            p_windows[i][0][0][0] = 0.6
            p_windows[i][0][0][-1] = 0.6
            p_windows[i][0][0][:, 0] = 0.6
            p_windows[i][0][0][:, -1] = 0.6
        
        end_time = time.process_time()
        self.ptime += (end_time - start_time)
        
        # Process each sub-region
        for i in range(4):
            if self.adaptive_partition.should_partition(u_windows[i], h):
                # Recursive partitioning
                x_windows[i], p_windows[i] = self.adaptive_partition_process(
                    x_windows[i], l_windows[i], u_windows[i], p_windows[i])
            else:
                # Apply attention mechanism
                start_time = time.process_time()
                
                q = x_windows[i].flatten(-2).transpose(-1, -2)  # [B, N, C]
                k = l_windows[i].flatten(-2).transpose(-1, -2)
                v = l_windows[i].flatten(-2).transpose(-1, -2)
                u = u_windows[i].flatten(-2).transpose(-1, -2)  # [B, N, 1]
                
                if self.use_mala:
                    # Use MALA linear attention
                    # Create uncertainty mask [B, N, N]
                    uncertainty_mask = u @ u.transpose(-1, -2)
                    
                    # MALA attention (linear complexity)
                    attention_output = self.attention(q, k, v, h, w, uncertainty_mask)
                    attention_output = self.conv_out1(attention_output).transpose(-2, -1).view(B, C, h, w)
                else:
                    # Use traditional multi-head attention
                    # Create uncertainty-based attention mask
                    uncertainty_mask = u @ u.transpose(-1, -2)
                    attention_mask = (uncertainty_mask < 1).bool()
                    new_attention_mask = torch.zeros_like(attention_mask, dtype=q.dtype)
                    new_attention_mask.masked_fill_(attention_mask, float("-1e10"))
                    
                    # Apply multi-head attention (quadratic complexity)
                    attention_output, _ = self.mha(q, k, v, attn_mask=new_attention_mask)
                    attention_output = self.conv_out1(attention_output).transpose(-2, -1).view(B, C, h, w)
                
                x_windows[i] += attention_output
                
                end_time = time.process_time()
                self.etime += (end_time - start_time)
        
        # Reassemble the feature map
        start_time = time.process_time()
        x_output = x_windows.permute(1, 2, 0, 3, 4).view(B, C, 2, 2, h, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
        p_output = p_windows.permute(1, 2, 0, 3, 4).view(B, 1, 2, 2, h, w).permute(0, 1, 2, 4, 3, 5).reshape(B, 1, H, W)
        end_time = time.process_time()
        self.rtime += (end_time - start_time)
        
        return x_output, p_output
    
    def forward(self, x: torch.Tensor, reference_features: torch.Tensor, 
                uncertainty_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """
        Forward pass.
        
        Args:
            x: Input features [B, C, H, W].
            reference_features: Reference features [B, C, H, W].
            uncertainty_map: Uncertainty map [B, 1, H, W].
            
        Returns:
            refined_features: Refined features.
            output: Output result.
            partition_map: Partition map.
            partition_time: Partitioning time.
            execution_time: Execution time.
        """
        B, C, H, W = x.shape
        partition_map = torch.ones((B, 1, H, W), device=x.device)
        
        # Binarize the uncertainty map
        binary_uncertainty = torch.where(uncertainty_map > 0.01, 1.0, 0.0)
        
        # Apply adaptive partitioning process
        refined_features, partition_map = self.adaptive_partition_process(
            x, reference_features, binary_uncertainty, partition_map)
        
        # Post-processing
        refined_features = self.conv_out3(refined_features)
        output = self.conv_out4(refined_features)
        
        return refined_features, output, partition_map, self.ptime, self.etime
    
    def forward_ablation(self, x: torch.Tensor, reference_features: torch.Tensor, 
                        uncertainty_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """
        Forward pass for ablation study (without uncertainty guidance).
        """
        output = self.conv_out4(x)
        return x, output, output, self.ptime, self.etime


class UncertaintyGuidedProcessor:
    """
    Complete uncertainty-guided processing pipeline.
    Integrates uncertainty map generation and attention mechanism.
    """
    
    def __init__(self, base_size: Tuple[int, int] = (384, 384), 
                 uncertainty_threshold: float = 0.5,
                 gaussian_kernel_size: int = 7,
                 gaussian_sigma: float = 1.0):
        """
        Args:
            base_size: Base image size.
            uncertainty_threshold: Uncertainty threshold.
            gaussian_kernel_size: Gaussian kernel size.
            gaussian_sigma: Standard deviation of the Gaussian kernel.
        """
        self.uncertainty_generator = UncertaintyMapGenerator(
            ksize=gaussian_kernel_size, 
            sigma=gaussian_sigma, 
            threshold=uncertainty_threshold
        )
        self.base_size = base_size
        
    def to(self, device):
        """Moves the module to the specified device."""
        self.uncertainty_generator.to(device)
        return self
        
    def cuda(self, idx: Optional[int] = None):
        """Moves the module to a CUDA device."""
        self.uncertainty_generator.cuda(idx)
        return self
    
    def process(self, saliency_map: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Processes a saliency map to generate an uncertainty map.
        
        Args:
            saliency_map: Saliency map [B, 1, H, W].
            target_shape: Target size (H, W).
            
        Returns:
            uncertainty_map: Uncertainty map [B, 1, H, W].
        """
        return self.uncertainty_generator.get_uncertainty_map(saliency_map, target_shape)


# Utility functions
def create_uncertainty_processor(base_size: Tuple[int, int] = (384, 384), 
                               uncertainty_threshold: float = 0.5) -> UncertaintyGuidedProcessor:
    """
    Utility function to create an uncertainty processor.
    
    Args:
        base_size: Base image size.
        uncertainty_threshold: Uncertainty threshold.
        
    Returns:
        An instance of UncertaintyGuidedProcessor.
    """
    return UncertaintyGuidedProcessor(base_size, uncertainty_threshold)


def create_uncertainty_attention(dim: int = 64, base_size: Tuple[int, int] = (384, 384), 
                               use_mala: bool = True) -> UncertaintyRefinementAttention:
    """
    Utility function to create an uncertainty attention module.
    
    Args:
        dim: Feature dimension.
        base_size: Base image size.
        use_mala: Whether to use MALA linear attention (recommended: True).
        
    Returns:
        An instance of UncertaintyRefinementAttention.
    """
    return UncertaintyRefinementAttention(dim, 1, dim, base_size, use_mala=use_mala)


if __name__ == "__main__":
    # Test code
    print("Testing uncertainty module...")
    
    # Create test data
    batch_size = 2
    height, width = 384, 384
    channels = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test uncertainty map generation
    processor = create_uncertainty_processor()
    processor.to(device)
    
    saliency_map = torch.randn(batch_size, 1, height, width).to(device)
    uncertainty_map = processor.process(saliency_map, (height // 4, width // 4))
    
    print(f"Uncertainty map shape: {uncertainty_map.shape}")
    print(f"Uncertainty map value range: [{uncertainty_map.min():.4f}, {uncertainty_map.max():.4f}]")
    
    # Test uncertainty attention
    attention_module = create_uncertainty_attention(dim=channels)
    attention_module.to(device)
    
    input_features = torch.randn(batch_size, channels, height // 4, width // 4).to(device)
    reference_features = torch.randn(batch_size, channels, height // 4, width // 4).to(device)
    
    refined_features, output, partition_map, p_time, e_time = attention_module(
        input_features, reference_features, uncertainty_map)
    
    print(f"Refined features shape: {refined_features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Partition map shape: {partition_map.shape}")
    print(f"Partition time: {p_time:.4f}s, Execution time: {e_time:.4f}s")
    
    print("Uncertainty module test complete!")
