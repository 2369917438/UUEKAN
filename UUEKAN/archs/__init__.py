"""
This package contains UUEKAN: a U-Net architecture combining EKAN and an uncertainty module (with integrated MALA linear attention).

Usage example:
    from archs import UUEKAN
    
    # Create UUEKAN model (recommended, with integrated MALA linear attention)
    model = UUEKAN(num_classes=1, img_size=256, embed_dims=[256, 320, 512], use_uncertainty=True)
    
"""

from .UUEKAN import UUEKAN

__all__ = [
    'UUEKAN',
]

__version__ = '1.0.0'

