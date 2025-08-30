"""
FF3D dataset for single object overfitting using canonical views.
Adapts the data format from test_gaussian_overfit.py to work with Splatter Image framework.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.general_utils import matrix_to_quaternion
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World, focal2fov
from .shared_dataset import SharedDataset


class FF3DDataset(SharedDataset):
    """
    Dataset for single object overfitting on FF3D canonical views.
    
    This dataset loads the canonical view data format from test_gaussian_overfit.py
    and adapts it to work with the Splatter Image training framework.
    """
    
    def __init__(self, cfg, dataset_name="train"):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        
        # Path to the object directory containing canonical views
        self.obj_dir = Path(cfg.data.obj_dir)
        if not self.obj_dir.exists():
            raise FileNotFoundError(f"Object directory not found: {self.obj_dir}")
            
        # Load canonical views metadata
        self.views, self.H, self.W = self.load_canonical_views()
        print(f"✅ Loaded {len(self.views)} canonical views ({self.H}x{self.W})")
        
        # Set up projection matrix
        self.projection_matrix = getProjectionMatrix(
            znear=self.cfg.data.znear, 
            zfar=self.cfg.data.zfar,
            fovX=cfg.data.fov * 2 * np.pi / 360, 
            fovY=cfg.data.fov * 2 * np.pi / 360
        ).transpose(0,1)
                
        # Simple train/test split: test_indices from config, everything else is training
        self.test_indices = cfg.data.test_indices
        self.train_indices = [i for i in range(len(self.views)) if i not in self.test_indices]
        
        print(f"Train indices: {self.train_indices[:10]}{'...' if len(self.train_indices) > 10 else ''}")
        print(f"Test indices: {self.test_indices}")

    def load_image(self, path: Path) -> np.ndarray:
        """Load RGB image and normalize to [0, 1]."""
        with Image.open(path) as im:
            # Resize to training resolution
            im = im.resize((self.cfg.data.training_resolution, self.cfg.data.training_resolution))
            arr = np.array(im)
        return arr.astype(np.float32) / 255.0

    def load_depth(self, path: Path) -> np.ndarray:
        """Load depth map in meters."""
        with Image.open(path) as im:
            # Resize to training resolution  
            im = im.resize((self.cfg.data.training_resolution, self.cfg.data.training_resolution))
            arr = np.array(im)
        if arr.dtype == np.uint16:
            depth_m = arr.astype(np.float32) / 1000.0
        elif arr.dtype in (np.float32, np.float64):
            depth_m = arr.astype(np.float32)
        else:
            raise RuntimeError(f"Unexpected depth dtype {arr.dtype} in {path}")
        return depth_m

    def load_mask(self, path: Path) -> np.ndarray:
        """Load binary mask."""
        with Image.open(path) as im:
            if im.mode != 'L':
                im = im.convert('L')
            # Resize to training resolution
            im = im.resize((self.cfg.data.training_resolution, self.cfg.data.training_resolution))
            arr = np.array(im)
        mask = (arr > 0).astype(np.float32)
        return mask

    def fix_metadata_path(self, broken_path: str) -> Path:
        """Fix incorrect path prefix in metadata."""
        return self.obj_dir / Path(broken_path).name

    def load_canonical_views(self) -> Tuple[List[Dict], int, int]:
        """Load all canonical view data from metadata."""
        views: List[Dict] = []
        metadata_path = self.obj_dir / 'canonical_views_metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        print(f"Loading canonical views from {metadata_path}")
        data = json.loads(metadata_path.read_text())
        Hc = int(data['canonical_img_H'])
        Wc = int(data['canonical_img_W'])
        view_list = data.get('views', [])
        
        for i, view_data in enumerate(view_list):
            rgb_path = self.fix_metadata_path(view_data['rgb_path'])
            depth_path = self.fix_metadata_path(view_data['depth_path'])
            mask_path = self.fix_metadata_path(view_data['mask_path'])
            
            # Check if files exist
            if not all(p.exists() for p in [rgb_path, depth_path, mask_path]):
                print(f"⚠️ Skipping view {i}: missing files")
                continue
            
            rgb = self.load_image(rgb_path)  # [H, W, 3]
            depth_m = self.load_depth(depth_path)  # [H, W]
            mask = self.load_mask(mask_path)  # [H, W]
            
            K = np.array(view_data['K'], dtype=np.float32)
            T_o2v = np.array(view_data['T_o2v'], dtype=np.float32)

            # Use world-to-camera (object-to-view) directly, matching other datasets' convention
            # Other datasets pass R = (w2c[:3, :3]).T and t = w2c[:3, 3] into getWorld2View2/getView2World
            w2c = T_o2v  # world/object -> camera/view
            R = w2c[:3, :3].T
            t = w2c[:3, 3]
            
            views.append({
                'rgb': rgb,  # [H, W, 3] 
                'depth': depth_m,  # [H, W]
                'mask': mask,  # [H, W]
                'K': K,  # [3, 3]
                'R': R,  # [3, 3] - camera rotation (world to camera), stored transposed as expected by utils
                't': t,  # [3] - camera translation (world to camera)
                'T_o2v': w2c,  # [4, 4] - original world-to-camera transform
            })
        
        return views, Hc, Wc

    def __len__(self):
        # For overfitting, we treat this as a single "object" dataset
        # But we can iterate through it multiple times per epoch
        return 1

    def __getitem__(self, index):
        """
        Returns ALL views in the canonical format expected by Splatter Image.
        
        Simple logic:
        - Always returns ALL 42 views 
        - Training loop decides which views to supervise on (train_indices)
        - Visualization can access any view (including test_indices)
        """
        # Simple: load ALL views, let training loop decide what to use
        all_indices = list(range(len(self.views)))
        
        # Convert to tensors and create camera matrices
        gt_images = []
        world_view_transforms = []
        view_to_world_transforms = []
        full_proj_transforms = []
        camera_centers = []
        focals_pixels = []
        
        for idx in all_indices:
            view = self.views[idx]
            
            # RGB image
            rgb = torch.from_numpy(view['rgb']).permute(2, 0, 1)  # [3, H, W]
            gt_images.append(rgb)
            
            # Camera transforms - use the matrices from the view
            R = view['R']
            t = view['t']
            
            world_view_transform = torch.tensor(
                getWorld2View2(R, t, np.array([0.0, 0.0, 0.0]), 1.0)
            ).transpose(0, 1).float()
            
            view_to_world_transform = torch.tensor(
                getView2World(R, t, np.array([0.0, 0.0, 0.0]), 1.0) 
            ).transpose(0, 1).float()
            
            # Per-view projection: compute FOV from intrinsics and training resolution
            FovX = focal2fov(K[0, 0], self.cfg.data.training_resolution)
            FovY = focal2fov(K[1, 1], self.cfg.data.training_resolution)
            projection_matrix = getProjectionMatrix(
                znear=self.cfg.data.znear,
                zfar=self.cfg.data.zfar,
                fovX=FovX,
                fovY=FovY,
            ).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
                projection_matrix.unsqueeze(0))).squeeze(0)
            
            camera_center = world_view_transform.inverse()[3, :3]
            
            # Extract focal lengths from K matrix for this view
            K = view['K']
            fx = K[0, 0]  # Focal length in x
            fy = K[1, 1]  # Focal length in y
            focal_pixels = torch.tensor([fx, fy], dtype=torch.float32)
            
            world_view_transforms.append(world_view_transform)
            view_to_world_transforms.append(view_to_world_transform)
            full_proj_transforms.append(full_proj_transform)
            camera_centers.append(camera_center)
            focals_pixels.append(focal_pixels)
        
        # Stack all tensors - note: don't add batch dimension yet
        images_and_camera_poses = {
            "gt_images": torch.stack(gt_images),  # [N_views, 3, H, W]
            "world_view_transforms": torch.stack(world_view_transforms),  # [N_views, 4, 4]  
            "view_to_world_transforms": torch.stack(view_to_world_transforms),  # [N_views, 4, 4]
            "full_proj_transforms": torch.stack(full_proj_transforms),  # [N_views, 4, 4]
            "camera_centers": torch.stack(camera_centers),  # [N_views, 3]
            "focals_pixels": torch.stack(focals_pixels),  # [N_views, 2]
        }
        
        # Make poses relative to first camera (expects no batch dimension)
        images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)
        
        # Add quaternion representation for camera rotations  
        images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(
            images_and_camera_poses["view_to_world_transforms"]
        )
        
        return images_and_camera_poses
