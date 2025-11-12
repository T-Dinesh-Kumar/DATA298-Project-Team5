# Notebooks Directory

This directory contains Jupyter notebooks for training, evaluation, and experimentation with the three text-to-video models.

## Structure

```
notebooks/
├── modelscope/
│   ├── model_1_production_code_google_colab.ipynb       # Training code
│   └── model_1_Video_Evaluation_Metrics.ipynb           # Evaluation metrics
├── cogvideox/
│   └── model_2_Fashion_Dataset_final_p1.ipynb           # Training + Evaluation
└── animatediff/
    └── model_3_Final-Model3.ipynb                       # Training + Evaluation
```

## Notebooks Overview

### Model 1: ModelScope
- **Training Notebook**: Complete training pipeline for 10,000 videos
- **Evaluation Notebook**: Comprehensive video quality metrics
- **Hardware**: A100 80GB
- **Time**: 9.48 hours training

### Model 2: CogVideoX-2B
- **Combined Notebook**: Training and evaluation in one file
- **Achievement**: 82.2% quality improvement
- **Hardware**: A100 80GB
- **Time**: 62 minutes fine-tuning

### Model 3: AnimateDiff LoRA
- **Combined Notebook**: LoRA training and evaluation
- **Achievement**: 30.2% temporal consistency improvement
- **Hardware**: H200 140GB (or A100 40GB+)
- **Time**: 8 minutes training

## Running Notebooks

### Local Jupyter
```bash
jupyter notebook
# Navigate to the notebook you want to run
```

### Google Colab
1. Upload notebook to Google Drive
2. Open with Google Colab
3. Change runtime to GPU (T4/V100/A100)
4. Run cells sequentially

### JupyterLab
```bash
jupyter lab
# Navigate to notebooks/ directory
```

## GPU Requirements

| Model | Minimum GPU | Recommended | VRAM |
|-------|-------------|-------------|------|
| ModelScope | A100 40GB | A100 80GB | 75GB |
| CogVideoX-2B | A100 40GB | A100 80GB | 76GB |
| AnimateDiff | RTX 3090 24GB | H200 140GB | 45GB |

## Notes

- **Checkpoints**: Notebooks save checkpoints to `outputs/` directory
- **Logging**: TensorBoard logs available in respective output directories
- **Interruption**: All notebooks support checkpoint resuming
- **Memory**: Enable gradient checkpointing if CUDA OOM errors occur

## Quick Start

Choose based on your goal:

1. **Just want to see results?**
   - Start with Model 3 (AnimateDiff) - fastest (8 minutes)

2. **Want best quality?**
   - Try Model 2 (CogVideoX) - 82.2% improvement (62 minutes)

3. **Want to work with large datasets?**
   - Use Model 1 (ModelScope) - 10K videos (9.48 hours)

## Documentation

For detailed information, see:
- [Model 1 Docs](../docs/MODEL1.md)
- [Model 2 Docs](../docs/MODEL2.md)
- [Model 3 Docs](../docs/MODEL3.md)
- [Setup Guide](../docs/SETUP.md)
- [Quick Start](../docs/QUICK_START.md)
