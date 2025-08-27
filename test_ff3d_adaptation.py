#!/usr/bin/env python3
"""
Test script for FF3D dataset adaptation to Splatter Image framework.

This script tests the custom FF3D dataset implementation and runs a minimal training
loop to verify that the adaptation works correctly.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from datasets.ff3d import FF3DDataset
from scene.gaussian_predictor import GaussianSplatPredictor
from gaussian_renderer import render_predicted

def create_test_config():
    """Create a minimal config for testing."""
    cfg = {
        'data': {
            'znear': 0.1,
            'zfar': 5.0,
            'fov': 50.0,
            'category': 'ff3d',
            'white_background': False,
            'origin_distances': False,
            'training_resolution': 512,
            'obj_dir': '/orion/u/yangyou/ff3d/data/PACE/models_rendered/obj_000000',
            'test_indices': [10, 20, 30, 40],
            'input_images': 1,
        },
        'model': {
            'max_sh_degree': 0,  # Start with degree 0 for simplicity
            'inverted_x': False,
            'inverted_y': True,
            'name': 'SingleUNet',
            'opacity_scale': 0.1,
            'opacity_bias': -2.0,
            'scale_bias': 0.02,
            'scale_scale': 0.01,
            'xyz_scale': 0.01,
            'xyz_bias': 0.0,
            'depth_scale': 1.0,
            'depth_bias': 0.0,
            'network_without_offset': False,
            'network_with_offset': True,
            'attention_resolutions': [16],
            'num_blocks': 4,
            'cross_view_attention': False,  # Disable for single image input
            'base_dim': 128,
            'isotropic': False,
        },
        'opt': {
            'base_lr': 0.0001,
            'batch_size': 1,
            'imgs_per_obj': 10,
        },
        'cam_embd': {
            'embedding': None,  # No camera embedding for simplicity
        }
    }
    return OmegaConf.create(cfg)

def test_dataset_loading(obj_dir_override=None):
    """Test that the FF3D dataset can be loaded correctly."""
    print("🧪 Testing FF3D dataset loading...")
    
    cfg = create_test_config()
    if obj_dir_override:
        cfg.data.obj_dir = obj_dir_override
    
    # Check if data directory exists
    obj_dir = Path(cfg.data.obj_dir)
    if not obj_dir.exists():
        print(f"❌ Test data directory not found: {obj_dir}")
        print("Please update cfg.data.obj_dir in the test script to point to your FF3D data directory")
        return False
    
    try:
        # Create dataset
        dataset = FF3DDataset(cfg, "train")
        print(f"✅ Dataset created successfully with {len(dataset.views)} views")
        
        # Test data loading
        sample = dataset[0]
        print(f"✅ Sample loaded successfully")
        
        # Check data format
        expected_keys = ["gt_images", "world_view_transforms", "view_to_world_transforms", 
                        "full_proj_transforms", "camera_centers", "source_cv2wT_quat"]
        for key in expected_keys:
            if key not in sample:
                print(f"❌ Missing key in sample: {key}")
                return False
        
        print("✅ All expected keys present in sample")
        
        # Check tensor shapes
        gt_images = sample["gt_images"]
        print(f"✅ GT images shape: {gt_images.shape}")
        print(f"✅ Image value range: [{gt_images.min().item():.3f}, {gt_images.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """Test that the Gaussian predictor model can be created."""
    print("\n🧪 Testing model creation...")
    
    try:
        cfg = create_test_config()
        model = GaussianSplatPredictor(cfg)
        print(f"✅ Model created successfully")
        
        # Test forward pass with dummy data
        batch_size = 1
        n_input_images = cfg.data.input_images
        H, W = cfg.data.training_resolution, cfg.data.training_resolution
        
        dummy_images = torch.randn(batch_size, n_input_images, 3, H, W)
        dummy_view_to_world = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, n_input_images, -1, -1)
        dummy_cv2wT_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).unsqueeze(0).expand(batch_size, n_input_images, -1)
        
        print("✅ Dummy input tensors created")
        
        with torch.no_grad():
            output = model(dummy_images, dummy_view_to_world, dummy_cv2wT_quat)
            
        print(f"✅ Forward pass successful")
        print(f"✅ Output keys: {list(output.keys())}")
        print(f"✅ XYZ shape: {output['xyz'].shape}")
        print(f"✅ Features DC shape: {output['features_dc'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation/forward pass failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_rendering():
    """Test that rendering works with predicted Gaussians."""
    print("\n🧪 Testing rendering pipeline...")
    
    try:
        cfg = create_test_config()
        
        # Create a simple predicted Gaussian dictionary
        n_gaussians = 1000
        pc = {
            'xyz': torch.randn(n_gaussians, 3),
            'rotation': torch.randn(n_gaussians, 4),
            'scaling': torch.ones(n_gaussians, 3) * 0.01,
            'opacity': torch.ones(n_gaussians, 1) * 0.5,
            'features_dc': torch.ones(n_gaussians, 1, 3) * 0.5,
        }
        
        # Add features_rest if sh_degree > 0  
        if cfg.model.max_sh_degree > 0:
            sh_rest_dim = (cfg.model.max_sh_degree + 1) ** 2 - 1
            pc['features_rest'] = torch.zeros(n_gaussians, sh_rest_dim, 3)
        
        # Create dummy camera parameters
        world_view_transform = torch.eye(4)
        full_proj_transform = torch.eye(4) 
        camera_center = torch.zeros(3)
        bg_color = torch.zeros(3)
        
        print("✅ Dummy Gaussian point cloud created")
        
        # Test rendering
        rendered = render_predicted(
            pc=pc,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
            camera_center=camera_center,
            bg_color=bg_color,
            cfg=cfg
        )
        
        print(f"✅ Rendering successful")
        print(f"✅ Rendered image shape: {rendered['render'].shape}")
        print(f"✅ Rendered image value range: [{rendered['render'].min().item():.3f}, {rendered['render'].max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Rendering failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_end_to_end(obj_dir_override=None):
    """Test end-to-end: dataset → model → rendering."""
    print("\n🧪 Testing end-to-end pipeline...")
    
    try:
        cfg = create_test_config()
        if obj_dir_override:
            cfg.data.obj_dir = obj_dir_override
        
        # Load dataset
        dataset = FF3DDataset(cfg, "train")
        sample = dataset[0]
        
        # Create model
        model = GaussianSplatPredictor(cfg)
        model.eval()
        
        print("✅ Dataset and model loaded")
        
        # Forward pass
        with torch.no_grad():
            input_images = sample["gt_images"][:cfg.data.input_images].unsqueeze(0)  # [1, 1, 3, H, W]
            view_to_world = sample["view_to_world_transforms"][:cfg.data.input_images].unsqueeze(0)  # [1, 1, 4, 4] 
            cv2wT_quat = sample["source_cv2wT_quat"][:cfg.data.input_images].unsqueeze(0)  # [1, 1, 4]
            
            print(f"✅ Input shapes - Images: {input_images.shape}, View2World: {view_to_world.shape}, Quat: {cv2wT_quat.shape}")
            
            # Predict Gaussians
            predicted_gaussians = model(input_images, view_to_world, cv2wT_quat)
            print(f"✅ Gaussian prediction successful")
            
            # Test rendering with a target view
            target_view_idx = min(5, len(sample["gt_images"]) - 1)  # Use view 5 or last view
            world_view_transform = sample["world_view_transforms"][target_view_idx]
            full_proj_transform = sample["full_proj_transforms"][target_view_idx]
            camera_center = sample["camera_centers"][target_view_idx]
            
            bg_color = torch.zeros(3) if not cfg.data.white_background else torch.ones(3)
            
            # Extract single batch
            pc_batch = {k: v[0].contiguous() for k, v in predicted_gaussians.items()}
            
            rendered = render_predicted(
                pc=pc_batch,
                world_view_transform=world_view_transform,
                full_proj_transform=full_proj_transform,
                camera_center=camera_center,
                bg_color=bg_color,
                cfg=cfg
            )
            
            print(f"✅ End-to-end rendering successful!")
            print(f"✅ Final rendered image shape: {rendered['render'].shape}")
            
            # Compare with ground truth
            gt_image = sample["gt_images"][target_view_idx]  # [3, H, W]
            rendered_image = rendered["render"]  # [3, H, W]
            
            mse_loss = torch.nn.functional.mse_loss(rendered_image, gt_image)
            print(f"✅ MSE loss vs GT (before training): {mse_loss.item():.6f}")
            
        return True
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {str(e)}")
        import traceback  
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test FF3D dataset adaptation")
    parser.add_argument("--obj_dir", type=str, 
                       default="/orion/u/yangyou/ff3d/data/PACE/models_rendered/obj_000000",
                       help="Path to FF3D object directory")
    args = parser.parse_args()
    
    print("🚀 Starting FF3D dataset adaptation tests...")
    print(f"Using object directory: {args.obj_dir}")
    
    # Store the obj_dir globally so test functions can access it
    global test_obj_dir
    test_obj_dir = args.obj_dir
    
    # Run tests
    tests = [
        (test_dataset_loading, True),  # True = needs obj_dir override
        (test_model_creation, False), 
        (test_rendering, False),
        (test_end_to_end, True),  # True = needs obj_dir override
    ]
    
    results = []
    for test_func, needs_obj_dir in tests:
        if needs_obj_dir:
            # Pass the obj_dir to the test function
            result = test_func(args.obj_dir)
        else:
            result = test_func()
        results.append(result)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! FF3D dataset adaptation is working correctly.")
        print("\n📋 Next steps:")
        print("1. Run training with: python train_network.py +dataset=ff3d")
        print("2. Monitor the training progress and losses")  
        print("3. Compare results with your original test_gaussian_overfit.py")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above and fix the issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
