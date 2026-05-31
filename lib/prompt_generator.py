"""
Differentiable Grounding-Aware Prompt Generator (GPG) for Enhanced_RRSIS_UOT.

Replaces the original non-differentiable GPG that used @torch.no_grad(),
torch.topk, and integer floor_div/modulo for coordinate extraction.

Key innovations:
    1. Spatial Soft-Argmax: Differentiable coordinate extraction via weighted
       average over a coordinate grid, replacing discrete topk + integer ops.
    2. Learnable Temperature: Controls softmax sharpness — starts soft (smooth
       gradients) and learns to sharpen during training.
    3. Refinement Conv: Small near-identity convolutional layer that learns to
       adjust OT heatmap peaks for better prompt placement.
    4. Iterative Gaussian Suppression: Differentiable multi-point extraction
       by suppressing already-selected regions with smooth Gaussian masks.

Gradient Flow:
    seg_loss → decoder → geometry_encoder → nn.Linear(2, d_model)
    → point coords (x,y) → spatial_soft_argmax → softmax(heatmap/τ)
    → heatmap = P_valid.sum(dim=-1) → P (OT transport plan)
    → sinkhorn(cost) → cost = 1 - cosine(img_proj, txt_proj)
    → img_proj, txt_proj (TRAINABLE)

Reference:
    Sun et al., "Integral Human Pose Regression", ECCV 2018.
    (Spatial soft-argmax / integral regression technique)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableGPG(nn.Module):
    """
    Differentiable Grounding-Aware Prompt Generator.

    Generates geometric point prompts from the OT transport plan with full
    gradient flow, enabling the segmentation loss to optimize prompt placement.

    Args:
        num_points: Maximum number of prompt points to extract (default: 5).
        initial_temperature: Starting softmax temperature (default: 1.0).
            Higher = softer attention (early training).
            Lower = sharper peaks (late training, approaches argmax).
        suppression_sigma: Gaussian suppression width in normalized coords
            (default: 0.15). Controls minimum spacing between points.
    """

    def __init__(self, num_points=5, initial_temperature=1.0, suppression_sigma=0.15):
        super().__init__()
        self.num_points = num_points
        self.suppression_sigma = suppression_sigma

        # Learnable log-temperature: exp(log_temperature) = actual temperature
        # Initialized to log(1.0) = 0.0 → temperature starts at 1.0
        self.log_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))

        # Small refinement conv: learns residual adjustment to heatmap peaks.
        # Near-identity initialization ensures minimal disruption at start.
        self.refine = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1, bias=True),
        )
        # Initialize near-zero so refine acts as identity + small residual
        nn.init.zeros_(self.refine[0].weight)
        nn.init.zeros_(self.refine[0].bias)
        nn.init.zeros_(self.refine[2].weight)
        nn.init.zeros_(self.refine[2].bias)

        print(f"[DifferentiableGPG] Initialized (num_points={num_points}, "
              f"temperature={initial_temperature:.1f}, sigma={suppression_sigma:.2f})")

    def _spatial_soft_argmax(self, heatmap, grid_x, grid_y):
        """
        Differentiable coordinate extraction via spatial soft-argmax.

        Computes the expected (x, y) coordinates as a weighted average
        over a pre-computed coordinate grid, using temperature-scaled
        softmax weights derived from the heatmap.

        Args:
            heatmap: (B, H, W) — activation map from OT plan.
            grid_x: (H, W) — pre-computed x-coordinate grid in [0, 1].
            grid_y: (H, W) — pre-computed y-coordinate grid in [0, 1].

        Returns:
            coords: (B, 2) — expected (x, y) coordinates in [0, 1].
        """
        B, H, W = heatmap.shape
        temperature = self.log_temperature.exp().clamp(min=0.01, max=10.0)

        # Flatten spatial dims and apply temperature-scaled softmax
        flat = heatmap.reshape(B, -1) / temperature  # (B, HW)
        weights = F.softmax(flat, dim=-1)  # (B, HW)
        weights = weights.view(B, H, W)  # (B, H, W)

        # Expected coordinates: weighted average over grid
        expected_x = (weights * grid_x.unsqueeze(0)).sum(dim=(1, 2))  # (B,)
        expected_y = (weights * grid_y.unsqueeze(0)).sum(dim=(1, 2))  # (B,)

        return torch.stack([expected_x, expected_y], dim=-1)  # (B, 2)

    def forward(self, P, text_mask, original_image_size, device):
        """
        Generate differentiable geometric prompts from OT transport plan.

        Args:
            P: (B, HW, seq) transport plan from the deepest OT aligner.
            text_mask: (B, seq) boolean mask (True = padding).
            original_image_size: int, size of the input image (e.g., 504).
            device: torch.device.

        Returns:
            points: (B, num_points, 2) absolute pixel coordinates (x, y).
            points_mask: (B, num_points) boolean validity mask.
            point_labels: (B, num_points) foreground labels (1 = fg).
        """
        B, HW, seq = P.shape
        H = W = int(math.sqrt(HW))

        # 1. Mask out padding tokens in the transport plan
        if text_mask is not None:
            # text_mask: True = padding → invert to get valid mask
            valid_mask = (~text_mask).unsqueeze(1).float()  # (B, 1, seq)
            P_valid = P * valid_mask
        else:
            P_valid = P

        # 2. Aggregate over sequence dimension → spatial heatmap
        heatmap = P_valid.sum(dim=-1)  # (B, HW)
        heatmap = heatmap.view(B, H, W)  # (B, H, W)

        # 3. Learnable refinement (near-identity init, learns residual)
        heatmap_refined = heatmap + self.refine(heatmap.unsqueeze(1)).squeeze(1)

        # 4. Pre-compute coordinate grids normalized to [0, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.0, 1.0, H, device=device, dtype=heatmap.dtype),
            torch.linspace(0.0, 1.0, W, device=device, dtype=heatmap.dtype),
            indexing='ij'
        )

        # 5. Extract K points via iterative soft-argmax with Gaussian suppression
        all_points = []
        current_heatmap = heatmap_refined

        for k in range(self.num_points):
            # Spatial soft-argmax on current (possibly suppressed) heatmap
            coord = self._spatial_soft_argmax(current_heatmap, grid_x, grid_y)  # (B, 2)
            all_points.append(coord)

            if k < self.num_points - 1:
                # Gaussian suppression around the extracted point
                cx = coord[:, 0].view(B, 1, 1)  # (B, 1, 1) — x center
                cy = coord[:, 1].view(B, 1, 1)  # (B, 1, 1) — y center

                # Squared distance from each grid cell to the extracted point
                dist_sq = (grid_x.unsqueeze(0) - cx) ** 2 + (grid_y.unsqueeze(0) - cy) ** 2

                # Differentiable suppression: 1 near point, 0 far away → invert
                sigma_sq = self.suppression_sigma ** 2
                suppression = 1.0 - torch.exp(-dist_sq / (2.0 * sigma_sq))

                # Apply suppression (differentiable element-wise multiply)
                current_heatmap = current_heatmap * suppression

        # 6. Stack all points: (B, K, 2) in normalized [0, 1] coords
        points_norm = torch.stack(all_points, dim=1)  # (B, num_points, 2)

        # 7. Scale to absolute image pixel coordinates
        #    SAM3 geometry encoder expects (x, y) in pixel space
        points = points_norm * float(original_image_size)  # (B, K, 2)

        # 8. Scale-Aware Prompting (SAP): adapt point count based on object size
        points_mask = torch.ones(B, self.num_points, dtype=torch.bool, device=device)
        point_labels = torch.ones(B, self.num_points, dtype=torch.long, device=device)

        # Detect tiny objects via active area ratio in the original heatmap
        with torch.no_grad():
            flat_hm = heatmap.view(B, -1)
            b_max = flat_hm.max(dim=1, keepdim=True)[0] + 1e-6
            norm_hm = flat_hm / b_max
            active_area = (norm_hm > 0.5).sum(dim=1).float() / float(H * W)

            for b in range(B):
                if active_area[b] < 0.01:
                    # Tiny object: keep only 1 center point to avoid bg noise
                    points_mask[b, 1:] = False
                    point_labels[b, 1:] = 0

        return points, points_mask, point_labels
