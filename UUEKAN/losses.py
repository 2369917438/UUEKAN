import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'UncertaintyLoss', 'UncertaintyRegularizationLoss', 
           'BoundaryLoss', 'DeepSupervisionLoss', 'CombinedLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - uses Sobel operator to extract boundaries and compute loss.
    Encourages the model to pay more attention to predictions in boundary regions.
    """
    def __init__(self):
        super().__init__()
        # Sobel operator for boundary detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, input, target):
        """
        Args:
            input: prediction logits [B, 1, H, W]
            target: ground truth labels [B, 1, H, W]
        """
        # Ensure Sobel operator is on the correct device
        if self.sobel_x.device != input.device:
            self.sobel_x = self.sobel_x.to(input.device)
            self.sobel_y = self.sobel_y.to(input.device)
        
        # Apply sigmoid to predictions
        input_prob = torch.sigmoid(input)
        
        # Extract predicted boundaries
        input_boundary_x = F.conv2d(input_prob, self.sobel_x, padding=1)
        input_boundary_y = F.conv2d(input_prob, self.sobel_y, padding=1)
        input_boundary = torch.sqrt(input_boundary_x ** 2 + input_boundary_y ** 2 + 1e-6)
        
        # Extract ground truth boundaries
        target_boundary_x = F.conv2d(target, self.sobel_x, padding=1)
        target_boundary_y = F.conv2d(target, self.sobel_y, padding=1)
        target_boundary = torch.sqrt(target_boundary_x ** 2 + target_boundary_y ** 2 + 1e-6)
        
        # Calculate boundary loss (MSE)
        boundary_loss = F.mse_loss(input_boundary, target_boundary)
        
        return boundary_loss


class UncertaintyRegularizationLoss(nn.Module):
    """
    Uncertainty Regularization Loss
    
    Objectives:
    1. In correctly predicted regions, encourage low uncertainty.
    2. In incorrectly predicted regions, encourage high uncertainty (allowing the model to express doubt).
    3. In boundary regions, naturally allow high uncertainty.
    """
    def __init__(self, boundary_weight=2.0):
        super().__init__()
        self.boundary_weight = boundary_weight
        # Sobel operator for boundary detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, prediction, target, uncertainty_map):
        """
        Args:
            prediction: prediction logits [B, 1, H, W]
            target: ground truth labels [B, 1, H, W]
            uncertainty_map: uncertainty map [B, 1, H, W], values in [0,1], higher means more uncertain.
        """
        # Ensure Sobel operator is on the correct device
        if self.sobel_x.device != prediction.device:
            self.sobel_x = self.sobel_x.to(prediction.device)
            self.sobel_y = self.sobel_y.to(prediction.device)
        
        # Get prediction probabilities
        pred_prob = torch.sigmoid(prediction)
        
        # Calculate prediction error
        prediction_error = torch.abs(pred_prob - target)  # [B, 1, H, W]
        
        # Extract boundary regions
        target_boundary_x = F.conv2d(target, self.sobel_x, padding=1)
        target_boundary_y = F.conv2d(target, self.sobel_y, padding=1)
        boundary_mask = torch.sqrt(target_boundary_x ** 2 + target_boundary_y ** 2 + 1e-6)
        boundary_mask = (boundary_mask > 0.1).float()  # Binarize boundary mask
        
        # Non-boundary region mask
        non_boundary_mask = 1.0 - boundary_mask
        
        # Loss 1: In correctly predicted non-boundary regions, penalize high uncertainty.
        # Correctly predicted regions should have low uncertainty.
        correct_mask = (prediction_error < 0.2).float() * non_boundary_mask
        correct_uncertainty_loss = (uncertainty_map * correct_mask).mean()
        
        # Loss 2: In incorrectly predicted non-boundary regions, encourage high uncertainty.
        # Incorrectly predicted regions should have high uncertainty (allowing the model to express doubt).
        incorrect_mask = (prediction_error >= 0.2).float() * non_boundary_mask
        incorrect_uncertainty_loss = ((1.0 - uncertainty_map) * incorrect_mask * prediction_error).mean()
        
        # Loss 3: In boundary regions, encourage moderate uncertainty.
        # Boundary regions naturally have high uncertainty, we want the uncertainty map to reflect this.
        boundary_uncertainty_target = 0.6  # Desired uncertainty level for boundary regions
        boundary_uncertainty_loss = F.mse_loss(
            uncertainty_map * boundary_mask,
            torch.ones_like(uncertainty_map) * boundary_uncertainty_target * boundary_mask
        )
        
        # Combined loss
        total_loss = (
            correct_uncertainty_loss + 
            0.5 * incorrect_uncertainty_loss + 
            self.boundary_weight * boundary_uncertainty_loss
        )
        
        return total_loss


class ThresholdRegularizationLoss(nn.Module):
    """
    Threshold Regularization Loss
    Prevents learnable thresholds from degenerating to extreme values.
    """
    def __init__(self, target_min=0.2, target_max=0.8):
        super().__init__()
        self.target_min = target_min
        self.target_max = target_max
    
    def forward(self, thresholds):
        """
        Args:
            thresholds: List of learnable thresholds, each should be in the range [0,1] after sigmoid.
        """
        total_loss = 0.0
        
        for threshold in thresholds:
            # Constrain threshold to [0,1] with sigmoid
            threshold_value = torch.sigmoid(threshold)
            
            # Penalize too small thresholds
            if threshold_value < self.target_min:
                total_loss += (self.target_min - threshold_value) ** 2
            
            # Penalize too large thresholds
            if threshold_value > self.target_max:
                total_loss += (threshold_value - self.target_max) ** 2
        
        return total_loss / len(thresholds)


class UncertaintyConsistencyLoss(nn.Module):
    """
    Uncertainty Consistency Loss
    Ensures that uncertainty predictions from multiple stages are consistent.
    Deeper uncertainty should guide shallower uncertainty.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, uncertainty_maps):
        """
        Args:
            uncertainty_maps: List of uncertainty maps from multiple stages.
                             From deep to shallow: [stage4, stage3, stage2, stage1]
        """
        if len(uncertainty_maps) < 2:
            return torch.tensor(0.0, device=uncertainty_maps[0].device)
        
        total_loss = 0.0
        
        # Consistency between adjacent layers
        for i in range(len(uncertainty_maps) - 1):
            deeper_map = uncertainty_maps[i]
            shallower_map = uncertainty_maps[i + 1]
            
            # Upsample the deeper uncertainty map to the size of the shallower one
            if deeper_map.shape != shallower_map.shape:
                deeper_map_upsampled = F.interpolate(
                    deeper_map, 
                    size=shallower_map.shape[2:], 
                    mode='bilinear', 
                    align_corners=True
                )
            else:
                deeper_map_upsampled = deeper_map
            
            # Consistency loss: the shallower uncertainty should be similar to the deeper one
            consistency_loss = F.mse_loss(shallower_map, deeper_map_upsampled)
            total_loss += consistency_loss
        
        return total_loss / (len(uncertainty_maps) - 1)


class UncertaintyLoss(nn.Module):
    """
    Comprehensive Uncertainty Loss
    Integrates all uncertainty-related losses.
    """
    def __init__(self, 
                 use_regularization=True,
                 use_threshold_reg=True,
                 use_consistency=True,
                 reg_weight=0.1,
                 threshold_reg_weight=0.01,
                 consistency_weight=0.05):
        super().__init__()
        
        self.use_regularization = use_regularization
        self.use_threshold_reg = use_threshold_reg
        self.use_consistency = use_consistency
        
        self.reg_weight = reg_weight
        self.threshold_reg_weight = threshold_reg_weight
        self.consistency_weight = consistency_weight
        
        if use_regularization:
            self.uncertainty_reg_loss = UncertaintyRegularizationLoss()
        
        if use_threshold_reg:
            self.threshold_reg_loss = ThresholdRegularizationLoss()
        
        if use_consistency:
            self.consistency_loss = UncertaintyConsistencyLoss()
    
    def forward(self, prediction, target, uncertainty_maps, learnable_thresholds=None):
        """
        Args:
            prediction: Main prediction output [B, 1, H, W]
            target: Ground truth labels [B, 1, H, W]
            uncertainty_maps: List of uncertainty maps or a single uncertainty map.
            learnable_thresholds: List of learnable thresholds (optional).
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Ensure uncertainty_maps is a list
        if not isinstance(uncertainty_maps, list):
            uncertainty_maps = [uncertainty_maps]
        
        # 1. Uncertainty regularization loss (for each stage)
        if self.use_regularization:
            reg_loss = 0.0
            for uncertainty_map in uncertainty_maps:
                # Resize uncertainty map to the same size as prediction
                if uncertainty_map.shape != prediction.shape:
                    uncertainty_map_resized = F.interpolate(
                        uncertainty_map,
                        size=prediction.shape[2:],
                        mode='bilinear',
                        align_corners=True
                    )
                else:
                    uncertainty_map_resized = uncertainty_map
                
                reg_loss += self.uncertainty_reg_loss(prediction, target, uncertainty_map_resized)
            
            reg_loss = reg_loss / len(uncertainty_maps)
            total_loss += self.reg_weight * reg_loss
            loss_dict['uncertainty_reg'] = reg_loss.item()
        
        # 2. Threshold regularization loss
        if self.use_threshold_reg and learnable_thresholds is not None:
            threshold_loss = self.threshold_reg_loss(learnable_thresholds)
            total_loss += self.threshold_reg_weight * threshold_loss
            loss_dict['threshold_reg'] = threshold_loss.item()
        
        # 3. Uncertainty consistency loss
        if self.use_consistency and len(uncertainty_maps) > 1:
            consistency_loss = self.consistency_loss(uncertainty_maps)
            total_loss += self.consistency_weight * consistency_loss
            loss_dict['uncertainty_consistency'] = consistency_loss.item()
        
        loss_dict['uncertainty_total'] = total_loss.item()
        
        return total_loss, loss_dict


class DeepSupervisionLoss(nn.Module):
    """Deep Supervision Loss - computes weighted loss for multiple auxiliary outputs."""
    def __init__(self, base_loss=None, weights=None):
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else BCEDiceLoss()
        # Default weights: main output has the highest weight, auxiliary outputs decrease.
        self.weights = weights if weights is not None else [1.0, 0.8, 0.6, 0.4]
    
    def forward(self, outputs, target):
        """
        Args:
            outputs: Can be a single output or a tuple of (main_output, [aux_outputs]).
            target: Target labels.
        """
        if isinstance(outputs, tuple):
            main_output, aux_outputs = outputs
            
            # Main output loss
            total_loss = self.weights[0] * self.base_loss(main_output, target)
            
            # Auxiliary output losses
            for i, aux_output in enumerate(aux_outputs):
                weight_idx = min(i + 1, len(self.weights) - 1)
                total_loss += self.weights[weight_idx] * self.base_loss(aux_output, target)
            
            return total_loss
        else:
            # If there is only a single output, compute the loss directly.
            return self.base_loss(outputs, target)


class CombinedLoss(nn.Module):
    """
    Combined Loss Function - for UUEKAN and UUNET
    
    Integrates:
    1. BCE + Dice Loss (main segmentation loss)
    2. Boundary Loss
    3. Deep Supervision
    4. Uncertainty-related losses (regularization, threshold, consistency)
    """
    def __init__(self, 
                 boundary_weight=0.2,
                 deep_supervision_weights=None,
                 use_boundary=True,
                 use_uncertainty_loss=True,
                 uncertainty_loss_weight=0.1,
                 uncertainty_config=None):
        super().__init__()
        
        self.use_boundary = use_boundary
        self.boundary_weight = boundary_weight
        self.use_uncertainty_loss = use_uncertainty_loss
        self.uncertainty_loss_weight = uncertainty_loss_weight
        
        # Base loss
        self.base_loss = BCEDiceLoss()
        
        # Boundary loss
        if self.use_boundary:
            self.boundary_loss = BoundaryLoss()
        
        # Deep supervision loss
        self.deep_supervision_loss = DeepSupervisionLoss(
            base_loss=self.base_loss,
            weights=deep_supervision_weights
        )
        
        # Uncertainty loss
        if self.use_uncertainty_loss:
            uncertainty_config = uncertainty_config or {}
            self.uncertainty_loss = UncertaintyLoss(**uncertainty_config)
    
    def forward(self, outputs, target, uncertainty_maps=None, learnable_thresholds=None):
        """
        Args:
            outputs: Can be a single output or a tuple of (main_output, [aux_outputs]).
            target: Target labels.
            uncertainty_maps: List of uncertainty maps (optional, for uncertainty loss).
                            If None, uncertainty loss is not computed.
            learnable_thresholds: List of learnable thresholds (optional).
        
        Returns:
            total_loss: Total loss.
            loss_dict: Detailed dictionary of individual losses.
        
        Note:
            - Compatible with standard UNet (without uncertainty module): pass only outputs and target.
            - Compatible with UUEKAN/UUNET: pass all parameters.
        """
        loss_dict = {}
        
        # 1. Deep supervision loss (includes main loss and auxiliary losses)
        ds_loss = self.deep_supervision_loss(outputs, target)
        total_loss = ds_loss
        loss_dict['segmentation'] = ds_loss.item()
        
        # Get the main output
        if isinstance(outputs, tuple):
            main_output = outputs[0]
        else:
            main_output = outputs
        
        # 2. Boundary loss (computed only on the main output)
        if self.use_boundary:
            boundary_loss = self.boundary_loss(main_output, target)
            total_loss = total_loss + self.boundary_weight * boundary_loss
            loss_dict['boundary'] = boundary_loss.item()
        
        # 3. Uncertainty loss (computed only if uncertainty_maps are provided)
        if self.use_uncertainty_loss and uncertainty_maps is not None:
            uncertainty_loss, uncertainty_loss_dict = self.uncertainty_loss(
                main_output, target, uncertainty_maps, learnable_thresholds
            )
            total_loss = total_loss + self.uncertainty_loss_weight * uncertainty_loss
            loss_dict.update(uncertainty_loss_dict)
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
