# FF3D Dataset Adaptation for Splatter Image Framework

This document describes how we adapted the Splatter Image codebase to work with FF3D single-object overfitting tasks, similar to what `test_gaussian_overfit.py` was attempting to do.

## Overview

The original `test_gaussian_overfit.py` script had poor performance when training a neural network to output Gaussians from a single image. We suspected this was due to:

1. **Suboptimal model architecture**: Simple CNN/transformer vs sophisticated U-Net
2. **Rendering differences**: Custom `gaussian_util.py` vs proven `diff-gaussian-rasterization`
3. **Training methodology**: Less refined data handling, initialization, regularization

The Splatter Image framework has demonstrated excellent performance on similar single-view → multi-view Gaussian reconstruction tasks, so we adapted it for FF3D overfitting.

## Key Adaptations Made

### 1. Custom Dataset Class (`datasets/ff3d.py`)

Created `FF3DDataset` that:
- Loads canonical view metadata from `canonical_views_metadata.json`
- Converts FF3D data format to Splatter Image's expected format
- Handles train/test splits for overfitting evaluation
- Resizes images to training resolution
- Converts coordinate systems appropriately

**Key differences from original datasets:**
- Single object focus (len=1) vs multiple objects
- Always uses view 0 as input, samples target views for supervision
- Handles FF3D's coordinate system conventions

### 2. Configuration File (`configs/dataset/ff3d.yaml`)

Provides FF3D-specific configuration:
```yaml
data:
  category: ff3d
  obj_dir: "/path/to/obj_000000"  # Single object directory
  training_resolution: 512
  test_indices: [10, 20, 30, 40]
  
model:
  # More conservative initialization for overfitting
  opacity_scale: 0.1
  xyz_scale: 0.01
  
opt:
  batch_size: 1     # Single object
  base_lr: 0.0001   # Lower LR for stable overfitting
```

### 3. Dataset Factory Integration

Updated `datasets/dataset_factory.py` to recognize `category: ff3d` and instantiate `FF3DDataset`.

## Usage Instructions

### Prerequisites

1. Install Splatter Image dependencies (see main README.md)
2. Have FF3D canonical view data structured like:
   ```
   obj_000000/
   ├── canonical_views_metadata.json
   ├── 0000_canonical_rgb.png
   ├── 0000_canonical_depth.png  
   ├── 0000_canonical_mask.png
   └── ... (more views)
   ```

### Running the Adaptation

1. **Test data structure** (optional but recommended):
   ```bash
   cd /path/to/splatter-image-for-ff3d
   python simple_test_ff3d.py
   ```

2. **Update config** to point to your data:
   Edit `configs/dataset/ff3d.yaml` and set:
   ```yaml
   data:
     obj_dir: "/path/to/your/obj_000000"
   ```

3. **Train the model**:
   ```bash
   python train_network.py +dataset=ff3d
   ```

4. **Monitor training**:
   - Check Weights & Biases dashboard for loss curves
   - Training should show decreasing loss over iterations
   - Model will save checkpoints periodically

### Expected Behavior

The adapted framework should:
- Load the single FF3D object successfully
- Train on view 0 → multiple target views  
- Show decreasing training loss (unlike original `test_gaussian_overfit.py`)
- Generate reasonable novel view renderings
- Leverage Splatter Image's proven architecture and training methodology

## Implementation Details

### Data Flow

1. `FF3DDataset.__getitem__()` loads canonical views
2. Always returns view 0 as input, samples others as targets
3. Converts coordinate systems to match Splatter Image conventions
4. Training loop uses `GaussianSplatPredictor` to predict Gaussians
5. `render_predicted()` renders novel views for loss computation
6. Standard L2/LPIPS losses applied

### Key Differences from Original Splatter Image

- **Single object**: Dataset length=1, focuses on one object
- **Fixed input view**: Always uses view 0, unlike random sampling
- **Coordinate handling**: Adapts FF3D's object-to-view transforms
- **Resolution matching**: Uses FF3D's native 512x512 resolution

### Coordinate System Notes

FF3D provides:
- `T_o2v`: Object-to-view transformation matrices
- Intrinsic matrices `K`
- Images in standard format

Splatter Image expects:
- World-to-view and view-to-world transformations
- Camera centers
- Quaternion representations for rotations

The dataset handles these conversions automatically.

## Comparison with Original Approach

| Aspect | `test_gaussian_overfit.py` | Splatter Image Adaptation |
|--------|----------------------------|---------------------------|
| Model Architecture | Simple CNN/Transformer | Sophisticated U-Net with attention |
| Rendering | Custom `gaussian_util.py` | Proven `diff-gaussian-rasterization` |  
| Data Handling | Manual tensor operations | Robust dataset framework |
| Training Loop | Custom implementation | Well-tested training pipeline |
| Initialization | Basic parameter init | Carefully tuned parameter initialization |
| Regularization | Limited | Built-in EMA, gradient clipping, etc. |

## Troubleshooting

**Common Issues:**

1. **Import errors**: Ensure all Splatter Image dependencies are installed
2. **Data path issues**: Check that `obj_dir` points to correct directory
3. **CUDA errors**: Ensure CUDA is available for Gaussian rasterization
4. **Memory issues**: Reduce `training_resolution` or `imgs_per_obj` if needed

**Debugging Steps:**

1. Run `simple_test_ff3d.py` to verify data structure
2. Check that metadata contains expected number of views
3. Verify image files exist and are readable
4. Monitor initial training iterations for obvious failures

## Next Steps

1. **Compare Results**: Run both original and adapted approaches, compare:
   - Training loss curves
   - Final rendering quality
   - Training stability

2. **Hyperparameter Tuning**: Adjust learning rates, model size, etc. based on results

3. **Extension**: Once basic overfitting works, could extend to:
   - Multiple objects
   - Different view sampling strategies
   - Integration with larger FF3D pipeline

## Files Created/Modified

- `datasets/ff3d.py` - Custom FF3D dataset class  
- `configs/dataset/ff3d.yaml` - FF3D configuration
- `datasets/dataset_factory.py` - Added FF3D dataset support
- `simple_test_ff3d.py` - Data structure verification
- `FF3D_ADAPTATION_README.md` - This documentation

The adaptation maintains the original Splatter Image training script (`train_network.py`) unchanged, demonstrating the modularity of the framework.
