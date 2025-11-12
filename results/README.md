# Results Directory

This directory stores generated videos, evaluation metrics, and training artifacts for all three models.

## Structure

```
results/
├── modelscope/
│   ├── samples/              # Generated video samples
│   ├── metrics/              # Evaluation results (JSON, CSV)
│   └── training_logs/        # TensorBoard logs
├── cogvideox/
│   ├── samples/              # Fashion video outputs
│   ├── metrics/              # Quality comparison data
│   └── training_logs/        # Training curves
└── animatediff/
    ├── samples/              # Anime video outputs
    ├── metrics/              # Temporal consistency results
    └── training_logs/        # LoRA training logs
```

## Important Notes

⚠️ **Large Files**: Video outputs are NOT committed to Git (see `.gitignore`)
⚠️ **Storage**: Results can be large (GB-TB depending on number of samples)
⚠️ **Backup**: Use cloud storage (Google Drive, S3, etc.) for archival

## What Goes Here

### samples/
Generated video files from training and inference:
- `.mp4` - Video outputs
- `.gif` - Short animation previews
- `.png` - Video frames or thumbnails

### metrics/
Evaluation results and statistics:
- `metrics.json` - Quantitative scores (FVD, IS, CLIP)
- `comparison.csv` - Before/after comparisons
- `temporal_consistency.json` - Frame-to-frame metrics

### training_logs/
TensorBoard event files:
- Training loss curves
- Validation metrics
- Learning rate schedules
- GPU utilization

## Key Results Summary

### Model 1: ModelScope
- **Final Loss**: 0.1036
- **Training Time**: 9.48 hours
- **Dataset**: 10,000 videos

### Model 2: CogVideoX-2B
- **Quality Improvement**: 82.2% (4.5/10 → 8.2/10)
- **Fabric Patterns**: +100%
- **Dress Fit**: +100%
- **Sleeve Details**: +125%
- **Training Time**: 62 minutes

### Model 3: AnimateDiff LoRA
- **Temporal Consistency**: +30.2%
- **Training Time**: 8 minutes
- **Parameters**: 16M (1% of base)

## Viewing Results

### TensorBoard
```bash
# Model 1
tensorboard --logdir results/modelscope/training_logs

# Model 2
tensorboard --logdir results/cogvideox/training_logs

# Model 3
tensorboard --logdir results/animatediff/training_logs
```

### Videos
Use any video player:
```bash
# Linux
vlc results/cogvideox/samples/fashion_001.mp4

# Mac
open results/cogvideox/samples/fashion_001.mp4

# Windows
start results/cogvideox/samples/fashion_001.mp4
```

### Metrics
```python
import json
import pandas as pd

# Load metrics
with open('results/cogvideox/metrics/metrics.json') as f:
    metrics = json.load(f)
    print(f"Quality improvement: {metrics['quality_improvement']}")

# Load comparison
df = pd.read_csv('results/cogvideox/metrics/comparison.csv')
print(df)
```

## Directory Management

### Clean up space
```bash
# Remove old checkpoints (keep final only)
rm -rf results/*/samples/checkpoint-*

# Remove large videos (keep metrics)
find results/ -name "*.mp4" -size +100M -delete
```

### Archive results
```bash
# Compress for archival
tar -czf results_backup.tar.gz results/

# Upload to cloud (example: Google Drive, S3)
rclone copy results/ gdrive:text-to-video-results/
```

## Best Practices

1. **Organize by experiment**: Create subdirectories for different runs
   ```
   results/cogvideox/samples/
   ├── exp_001_baseline/
   ├── exp_002_higher_lr/
   └── exp_003_more_data/
   ```

2. **Save metadata**: Include config files with results
   ```python
   import json
   import shutil

   # Save config with results
   shutil.copy('configs/cogvideox_config.yaml',
               'results/cogvideox/exp_001/config.yaml')
   ```

3. **Document experiments**: Add README to each experiment
   ```markdown
   # Experiment 001: Baseline
   - Date: 2024-11-10
   - Config: Higher learning rate (1e-4 → 5e-4)
   - Result: Faster convergence but less stable
   ```

## Sharing Results

### For Portfolio
1. Select best samples (5-10 videos per model)
2. Create highlight reel or GIF animations
3. Host on YouTube/Vimeo with project description

### For Paper/Presentation
1. Export metrics as tables (LaTeX, CSV)
2. Create comparison visualizations
3. Include training curves from TensorBoard

### For GitHub
1. Add representative GIFs to README
2. Link to cloud storage for full results
3. Include metrics in markdown tables

## Cloud Storage Recommendations

**For Academic/Portfolio:**
- Google Drive (free 15GB)
- YouTube (unlimited video hosting)
- GitHub Releases (up to 2GB per file)

**For Production:**
- AWS S3
- Google Cloud Storage
- Azure Blob Storage

## Need Help?

See documentation:
- [Setup Guide](../docs/SETUP.md)
- [Model Docs](../docs/)
- [Main README](../README.md)
