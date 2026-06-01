import torch
import torch.nn as nn
import torch.nn.functional as F

class SobelFilter(nn.Module):
    """
    Lightweight Sobel edge filter to extract exact boundary masks in PyTorch.
    """
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)
        
    def forward(self, x):
        # Replicate padding keeps edge features clean
        padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(padded, self.kernel_x)
        grad_y = F.conv2d(padded, self.kernel_y)
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        # Normalize and clamp to represent binary-like boundary probabilities
        return torch.clamp(magnitude, 0.0, 1.0)


class TextGuidedBoundaryLoss(nn.Module):
    """
    Text-Guided Boundary Loss (L_tbl).
    Weights standard BCE loss heavier at the edges of the referred object
    by calculating a spatial cross-attention map between pooled text features
    and high-res image features, and modulating Sobel edge masks with it.
    """
    def __init__(self, gamma: float = 3.0):
        super().__init__()
        self.gamma = gamma
        self.sobel = SobelFilter()
        
    def forward(self, pred_masks, gt_masks, text_feat, img_feat):
        """
        Args:
            pred_masks: [B, 1, H, W] raw logits from model.
            gt_masks: [B, 1, H, W] ground-truth binary masks.
            text_feat: [seq_len, B, C] or [B, seq_len, C] language features.
            img_feat: [B, C, H_img, W_img] high-resolution image features.
        """
        B, C, H, W = pred_masks.shape
        device = pred_masks.device
        
        # 1. Extract GT boundaries
        with torch.no_grad():
            boundary = self.sobel(gt_masks.float())  # [B, 1, H, W]
            
        # 2. Compute text-visual attention map
        # Pool text features over sequence dimension: [seq_len, B, C] -> [B, C]
        if text_feat.dim() == 3 and text_feat.shape[1] == B:
            text_pooled = text_feat.mean(dim=0)  # [B, C]
        elif text_feat.dim() == 3:
            text_pooled = text_feat.mean(dim=1)  # [B, C]
        else:
            text_pooled = text_feat
            
        # Normalize visual features and text features for cosine metric
        img_norm = F.normalize(img_feat, dim=1)
        text_norm = F.normalize(text_pooled, dim=-1)
        
        # Flatten image features for dot product: [B, C, H_img, W_img] -> [B, C, H_img * W_img]
        H_img, W_img = img_feat.shape[-2:]
        img_flat = img_norm.view(B, -1, H_img * W_img)
        
        # Dot product: [B, 1, C] x [B, C, HW] -> [B, 1, HW]
        attn = torch.bmm(text_norm.unsqueeze(1), img_flat).squeeze(1)  # [B, HW]
        attn = torch.sigmoid(attn).view(B, 1, H_img, W_img)  # Spatial attention
        
        # Upsample attention if visual features are lower resolution than pred_masks
        if (H_img, W_img) != (H, W):
            attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)
            
        # 3. Weighted BCE loss computation
        bce = F.binary_cross_entropy_with_logits(pred_masks, gt_masks.float(), reduction='none')
        # Focus on edges that are highly correlated with the text query
        weight = 1.0 + self.gamma * boundary * attn
        
        loss = (bce * weight).mean()
        return loss
