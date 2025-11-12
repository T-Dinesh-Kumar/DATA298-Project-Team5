# Models Directory

This directory contains model-specific code, architectures, and configurations for the three text-to-video generation models.

## Structure

```
models/
├── modelscope/       # Model 1: Human Action Video Generation
├── cogvideox/        # Model 2: Fashion Video Generation
└── animatediff/      # Model 3: Anime Video Generation (LoRA)
```

## Usage

Each model subdirectory contains:
- Model architecture implementations (if custom)
- Training utilities
- Inference scripts
- Model-specific configurations

## Quick Links

- **Model 1 Documentation**: [../docs/MODEL1.md](../docs/MODEL1.md)
- **Model 2 Documentation**: [../docs/MODEL2.md](../docs/MODEL2.md)
- **Model 3 Documentation**: [../docs/MODEL3.md](../docs/MODEL3.md)

## Notes

- **Checkpoints**: Trained model weights are NOT stored in Git (see `.gitignore`)
- **Download**: Use Hugging Face Hub or cloud storage for model weights
- **Configs**: See `configs/` directory for training configurations
