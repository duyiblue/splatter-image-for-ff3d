#!/usr/bin/env python3
"""
Visualization script for trained FF3D model results.

This script loads a trained Splatter Image model and creates the same
visualizations as test_gaussian_overfit.py for comparison.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from datasets.ff3d import FF3DDataset
from scene.gaussian_predictor import GaussianSplatPredictor
from gaussian_renderer import render_predicted


def load_trained_model(checkpoint_path: str, config_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    # Load config
    if config_path.endswith('.yaml'):
        cfg = OmegaConf.load(config_path)
    else:
        # Try to find config in the same directory
        config_dir = Path(checkpoint_path).parent / '.hydra' / 'config.yaml'
        if config_dir.exists():
            cfg = OmegaConf.load(config_dir)
        else:
            raise FileNotFoundError(f"Could not find config file. Please provide path to config.yaml")
    
    # Create model
    model = GaussianSplatPredictor(cfg).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    print(f"✅ Loaded trained model from {checkpoint_path}")
    print(f"✅ Training iteration: {checkpoint.get('iteration', 'unknown')}")
    print(f"✅ Best PSNR: {checkpoint.get('best_PSNR', 'unknown'):.3f}")
    
    return model, cfg


def create_train_test_comparison(model: torch.nn.Module, dataset: FF3DDataset, cfg, 
                               device: torch.device, output_dir: Path):
    """Create train/test view comparison similar to test_gaussian_overfit.py."""
    
    # Get sample data
    sample = dataset[0]
    
    # Select views for visualization (same logic as test_gaussian_overfit.py)
    train_indices = dataset.train_indices
    test_indices = dataset.test_indices
    
    vis_train_indices = train_indices[:4] if len(train_indices) >= 4 else train_indices
    vis_test_indices = test_indices[:4] if len(test_indices) >= 4 else test_indices
    
    print(f"Visualizing train views: {vis_train_indices}")
    print(f"Visualizing test views: {vis_test_indices}")
    
    num_train = len(vis_train_indices)
    num_test = len(vis_test_indices) 
    num_cols = max(num_train, num_test)
    
    fig, axes = plt.subplots(4, num_cols, figsize=(4 * num_cols, 12))
    if num_cols == 1:
        axes = axes.reshape(4, 1)
    
    with torch.no_grad():
        # Use view 0 as input (same as original approach)
        input_view_idx = 0
        input_images = sample["gt_images"][input_view_idx:input_view_idx+1].unsqueeze(0).to(device)  # [1, 1, 3, H, W]
        view_to_world = sample["view_to_world_transforms"][input_view_idx:input_view_idx+1].unsqueeze(0).to(device)  # [1, 1, 4, 4]
        cv2wT_quat = sample["source_cv2wT_quat"][input_view_idx:input_view_idx+1].unsqueeze(0).to(device)  # [1, 1, 4]
        
        # Predict Gaussians
        predicted_gaussians = model(input_images, view_to_world, cv2wT_quat)
        
        # Extract single batch
        pc_batch = {k: v[0].contiguous() for k, v in predicted_gaussians.items()}
        
        bg_color = (torch.zeros(3) if not cfg.data.white_background else torch.ones(3)).to(device)
        
        # Render training views
        for i in range(num_train):
            view_idx = vis_train_indices[i]
            gt_rgb = sample["gt_images"][view_idx].permute(1, 2, 0).cpu().numpy()
            axes[0, i].imshow(gt_rgb)
            axes[0, i].set_title(f'Train GT {view_idx}')
            axes[0, i].axis('off')
            
            # Render this view
            rendered = render_predicted(
                pc=pc_batch,
                world_view_transform=sample["world_view_transforms"][view_idx].to(device),
                full_proj_transform=sample["full_proj_transforms"][view_idx].to(device),
                camera_center=sample["camera_centers"][view_idx].to(device),
                bg_color=bg_color,
                cfg=cfg
            )
            
            ren_rgb = rendered["render"].permute(1, 2, 0).detach().cpu().numpy()
            axes[1, i].imshow(ren_rgb.clip(0, 1))
            axes[1, i].set_title(f'Train Render {view_idx}')  
            axes[1, i].axis('off')
        
        # Render test views
        for i in range(num_test):
            view_idx = vis_test_indices[i]
            gt_rgb = sample["gt_images"][view_idx].permute(1, 2, 0).cpu().numpy()
            axes[2, i].imshow(gt_rgb)
            axes[2, i].set_title(f'Test GT {view_idx}')
            axes[2, i].axis('off')
            
            # Render this view
            rendered = render_predicted(
                pc=pc_batch,
                world_view_transform=sample["world_view_transforms"][view_idx].to(device),
                full_proj_transform=sample["full_proj_transforms"][view_idx].to(device), 
                camera_center=sample["camera_centers"][view_idx].to(device),
                bg_color=bg_color,
                cfg=cfg
            )
            
            ren_rgb = rendered["render"].permute(1, 2, 0).detach().cpu().numpy()
            axes[3, i].imshow(ren_rgb.clip(0, 1))
            axes[3, i].set_title(f'Test Render {view_idx}')
            axes[3, i].axis('off')
    
    plt.suptitle('Splatter Image Results: Train (rows 1-2) vs Test (rows 3-4)', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'train_test_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Visualization saved to {output_dir}/train_test_comparison.png")


def compute_metrics(model: torch.nn.Module, dataset: FF3DDataset, cfg, device: torch.device):
    """Compute training and test metrics."""
    sample = dataset[0]
    
    train_indices = dataset.train_indices
    test_indices = dataset.test_indices
    
    with torch.no_grad():
        # Use view 0 as input
        input_view_idx = 0
        input_images = sample["gt_images"][input_view_idx:input_view_idx+1].unsqueeze(0).to(device)
        view_to_world = sample["view_to_world_transforms"][input_view_idx:input_view_idx+1].unsqueeze(0).to(device)
        cv2wT_quat = sample["source_cv2wT_quat"][input_view_idx:input_view_idx+1].unsqueeze(0).to(device)
        
        # Predict Gaussians
        predicted_gaussians = model(input_images, view_to_world, cv2wT_quat)
        pc_batch = {k: v[0].contiguous() for k, v in predicted_gaussians.items()}
        
        bg_color = (torch.zeros(3) if not cfg.data.white_background else torch.ones(3)).to(device)
        
        # Compute training metrics
        train_mse_list = []
        for view_idx in train_indices:
            gt_image = sample["gt_images"][view_idx].to(device)
            
            rendered = render_predicted(
                pc=pc_batch,
                world_view_transform=sample["world_view_transforms"][view_idx].to(device),
                full_proj_transform=sample["full_proj_transforms"][view_idx].to(device),
                camera_center=sample["camera_centers"][view_idx].to(device),
                bg_color=bg_color,
                cfg=cfg
            )
            
            mse = torch.nn.functional.mse_loss(rendered["render"], gt_image)
            train_mse_list.append(mse.item())
        
        # Compute test metrics  
        test_mse_list = []
        for view_idx in test_indices:
            gt_image = sample["gt_images"][view_idx].to(device)
            
            rendered = render_predicted(
                pc=pc_batch,
                world_view_transform=sample["world_view_transforms"][view_idx].to(device),
                full_proj_transform=sample["full_proj_transforms"][view_idx].to(device),
                camera_center=sample["camera_centers"][view_idx].to(device),
                bg_color=bg_color,
                cfg=cfg
            )
            
            mse = torch.nn.functional.mse_loss(rendered["render"], gt_image)
            test_mse_list.append(mse.item())
    
    train_mse_avg = np.mean(train_mse_list)
    test_mse_avg = np.mean(test_mse_list)
    
    print(f"📊 Final Results:")
    print(f"  Train MSE: {train_mse_avg:.6f} (averaged over {len(train_indices)} views)")
    print(f"  Test MSE:  {test_mse_avg:.6f} (averaged over {len(test_indices)} views)")
    
    return {
        'train_mse': train_mse_avg,
        'test_mse': test_mse_avg,
        'train_views': len(train_indices), 
        'test_views': len(test_indices),
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize trained FF3D Splatter Image model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint (.pth file)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (if not in checkpoint dir/.hydra/)")
    parser.add_argument("--output_dir", type=str, default="./ff3d_visualization",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Visualizing trained FF3D model...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Load trained model
        model, cfg = load_trained_model(args.checkpoint, args.config, device)
        
        # Create dataset
        dataset = FF3DDataset(cfg, "train")  # Use train to get all data
        
        # Create visualizations
        create_train_test_comparison(model, dataset, cfg, device, output_dir)
        
        # Compute metrics
        metrics = compute_metrics(model, dataset, cfg, device)
        
        # Save metrics
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n🎉 Visualization complete!")
        print(f"📁 Results saved to: {output_dir}")
        print(f"📊 Metrics: Train MSE={metrics['train_mse']:.6f}, Test MSE={metrics['test_mse']:.6f}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
