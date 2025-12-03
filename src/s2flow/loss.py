import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Literal

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


