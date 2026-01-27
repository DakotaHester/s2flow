import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from typing import Optional, Union, List, Literal

from .data.pca import PCAConvLayer
from .utils import get_device

def focal_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    alpha: Optional[Union[float, List[float]]] = None,
    gamma: float = 2.0,
    smooth: float = 1e-6,
    reduction: Literal['mean', 'sum', 'none'] = 'mean'
) -> torch.Tensor:
    """
    Functional interface for computing Focal Loss.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted probabilities with shape (batch_size, num_classes, ...).
    y_true : torch.Tensor
        Ground truth labels with shape (batch_size, ...).
    alpha : Optional[Union[float, List[float]]]
        Class weights for addressing class imbalance.
    gamma : float
        Focusing parameter to penalize hard examples.
    smooth : float
        Smoothing term to avoid instability during logarithmic operations.
    reduction : Literal['mean', 'sum', 'none']
        Specifies the reduction method for the loss output.
    Returns
    -------
    torch.Tensor
        Computed loss. The shape depends on the `reduction` parameter.
    """
     # Clamp predictions to prevent extreme values
    y_pred = torch.clamp(y_pred, smooth, 1.0 - smooth)
    assert y_true.max() < y_pred.shape[1], "y_true contains class indices out of range."
    
    # Convert labels to one-hot encoding
    num_classes = y_pred.shape[1]
    y_true = F.one_hot(y_true.long(), num_classes).permute(0, -1, *range(1, y_true.dim()))
    y_true = y_true.float()
    
    # Calculate focal loss with stable log
    log_prob = torch.log(y_pred)
    prob = torch.exp(log_prob)
    
    # Calculate focal term
    focal_term = torch.pow(1 - prob, gamma)
    
    # Combine terms
    focal_loss = -y_true * focal_term * log_prob
    
    # Apply class weights if specified
    if alpha is not None:
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.to(y_pred.device)
        else:
            alpha = torch.tensor([alpha] * num_classes).to(y_pred.device)
        focal_loss = alpha.view(1, -1, *([1] * (focal_loss.dim() - 2))) * focal_loss
    
    # Sum over spatial dimensions
    dims = tuple(range(2, y_true.dim()))
    focal_loss = torch.sum(focal_loss, dims)
    
    # Handle any remaining numerical instabilities
    focal_loss = torch.nan_to_num(focal_loss, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Apply reduction
    if reduction == 'mean':
        return torch.mean(focal_loss)
    elif reduction == 'sum':
        return torch.sum(focal_loss)
    else:  # 'none'
        return focal_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class semantic segmentation.
    https://arxiv.org/abs/1708.02002

    Attributes
    ----------
    alpha : Optional[Union[float, List[float]]]
        Class weights for addressing class imbalance.
    gamma : float
        Focusing parameter to penalize hard examples.
    smooth : float
        Smoothing term to avoid instability during logarithmic operations.
    reduction : Literal['mean', 'sum', 'none']
        Specifies the reduction method for the loss output.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, List[float]]] = None,
        gamma: float = 2.0,
        smooth: float = 1e-6,
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        """
        Initialize the FocalLoss class.

        Parameters
        ----------
        alpha : Optional[Union[float, List[float]]]
            Class weights for addressing class imbalance.
        gamma : float
            Focusing parameter for controlling penalization of hard examples.
        smooth : float
            Smoothing factor to avoid instability in logarithmic computations.
        reduction : Literal['mean', 'sum', 'none']
            Specifies the reduction method for the loss output.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
        
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Focal Loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted probabilities with shape (batch_size, num_classes, ...).
        y_true : torch.Tensor
            Ground truth labels with shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            Computed loss. The shape depends on the `reduction` parameter.
        """
        return focal_loss(
            y_pred,
            y_true,
            alpha=self.alpha,
            gamma=self.gamma,
            smooth=self.smooth,
            reduction=self.reduction
        )


class MultispectralPerceptualLoss(nn.Module):
    """
    Perceptual loss wrapper that projects multispectral input (e.g., 4-band) 
    to 3-band RGB using a PCA layer before computing VGG features.
    """
    def __init__(self, config):
        super(MultispectralPerceptualLoss, self).__init__()
        self.device = get_device()
        
        # Initialize PCA Layer for dimensionality reduction (4 -> 3)
        self.pca_layer = PCAConvLayer(config).to(self.device)
        self.k = config.get('metrics', {}).get('pca_lpips_k', 3.0)
        self.clamp = config.get('metrics', {}).get('pca_lpips_clamp', True)

        # Load VGG19 features
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        self.features = nn.ModuleList(list(vgg.features)).eval()
        
        # Layer indices for "conv1_2", "conv2_2", "conv3_4", "conv4_4", "conv5_4"
        # Adjusted slightly to match standard implementation of "before activation"
        self.layer_indices = {
            'conv1': 2, 'conv2': 7, 'conv3': 16, 'conv4': 25, 'conv5': 34
        }
        self.weights = {'conv1': 0.1, 'conv2': 0.1, 'conv3': 1.0, 'conv4': 1.0, 'conv5': 1.0}

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
        """
        Args:
            pred: (B, C_in, H, W)
            target: (B, C_in, H, W)
        """
        # 1. Project to 3-channel using PCA (differentiable)
        # Note: PCAConvLayer expects standard forward, ensuring gradients flow back to Generator
        x_feat = self.pca_layer(pred, k=self.k, clamp=self.clamp)
        y_feat = self.pca_layer(target, k=self.k, clamp=self.clamp)
        
        loss = 0
        current_layer = 0
        for name, index in sorted(self.layer_indices.items(), key=lambda item: item[1]):
            for i in range(current_layer, index):
                x_feat = self.features[i](x_feat)
                y_feat = self.features[i](y_feat)
            
            x_feat = self.features[index](x_feat)
            y_feat = self.features[index](y_feat)
            
            loss += self.weights[name] * F.l1_loss(x_feat, y_feat)
            current_layer = index + 1
            
        return loss