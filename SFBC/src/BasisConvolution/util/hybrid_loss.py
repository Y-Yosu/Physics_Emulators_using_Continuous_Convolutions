import torch
import torch.nn.functional as F


def compute_hybrid_loss(predictions, ground_truths, alpha=0.7, beta=0.3):
    """
    Compute hybrid loss combining MSE and correlation.
    
    Loss = α × MSE + β × (1 - correlation)
    
    Both terms optimize toward 0:
    - MSE = 0 when predictions perfectly match ground truth
    - (1 - correlation) = 0 when correlation = 1 (perfect correlation)
    
    Args:
        predictions: List of prediction tensors
        ground_truths: List of ground truth tensors  
        alpha: Weight for MSE loss (default: 0.7)
        beta: Weight for correlation loss (default: 0.3)
    
    Returns:
        dict: {
            'total_loss': Combined loss tensor,
            'mse_loss': Pure MSE component tensor,
            'correlation_loss': Pure correlation component tensor (1 - correlation),
            'raw_correlation': Average correlation value tensor,
            'individual_mse': List of individual MSE losses,
            'individual_correlations': List of individual correlation values
        }
    """
    # Calculate MSE losses
    mse_losses = [F.mse_loss(pred, gt) for pred, gt in zip(predictions, ground_truths)]
    mse_loss = torch.stack(mse_losses).mean()
    
    # Calculate correlations
    correlations = []
    for pred, gt in zip(predictions, ground_truths):
        # Flatten tensors for correlation calculation
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        
        # Calculate means
        pred_mean = pred_flat.mean()
        gt_mean = gt_flat.mean()
        
        # Center the data
        pred_centered = pred_flat - pred_mean
        gt_centered = gt_flat - gt_mean
        
        # Calculate Pearson correlation coefficient
        numerator = (pred_centered * gt_centered).sum()
        denominator = torch.sqrt((pred_centered**2).sum() * (gt_centered**2).sum())
        
        # Add small epsilon to avoid division by zero
        correlation = numerator / (denominator + 1e-8)
        
        # Clamp correlation to valid range [-1, 1] for numerical stability
        correlation = torch.clamp(correlation, -1.0, 1.0)
        
        correlations.append(correlation)
    
    # Average correlation across batch
    avg_correlation = torch.stack(correlations).mean()
    
    # Correlation loss: 1 - correlation
    # When correlation = 1 (perfect), loss = 0
    # When correlation = 0 (no correlation), loss = 1  
    # When correlation = -1 (anti-correlation), loss = 2
    correlation_loss = 1.0 - avg_correlation
    
    # Optional: Add scale penalty to prevent predictions from drifting
    # This helps when using correlation loss which is scale-invariant
    scale_penalty = 0.0
    for pred, gt in zip(predictions, ground_truths):
        pred_mean = pred.mean()
        gt_mean = gt.mean()
        # Penalize if mean prediction deviates from ground truth mean
        scale_penalty += ((pred_mean - gt_mean) / (gt_mean + 1e-8))**2
    scale_penalty = scale_penalty / len(predictions)
    
    # Combine losses with weights
    # Add small scale penalty to prevent correlation from allowing wrong scales
    gamma = 0.05  # Small weight for scale penalty
    total_loss = alpha * mse_loss + beta * correlation_loss + gamma * scale_penalty
    
    return {
        'total_loss': total_loss,
        'mse_loss': mse_loss,
        'correlation_loss': correlation_loss,
        'scale_penalty': scale_penalty,
        'raw_correlation': avg_correlation,
        'individual_mse': mse_losses,
        'individual_correlations': correlations
    }


def compute_psnr(predictions, ground_truths):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) for a batch.
    
    Args:
        predictions: List of prediction tensors
        ground_truths: List of ground truth tensors
        
    Returns:
        List of PSNR values
    """
    psnrs = []
    for pred, gt in zip(predictions, ground_truths):
        mse = F.mse_loss(pred, gt)
        max_val = gt.abs().max()
        psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse)
        psnrs.append(psnr)
    
    return psnrs
