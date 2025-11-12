# Setup Guide - Text-to-Video Generation Project

This guide will help you set up the environment and get started with the text-to-video generation models.

---

## 📋 Prerequisites

### Hardware Requirements

| Model | Minimum GPU | Recommended GPU | VRAM | Training Time |
|-------|-------------|-----------------|------|---------------|
| ModelScope | A100 40GB | A100 80GB | 75GB | 9.48 hours |
| CogVideoX-2B | A100 40GB | A100 80GB | 76GB | 62 minutes |
| AnimateDiff LoRA | RTX 3090 24GB | H200 140GB | 45GB | 8 minutes |

### Software Requirements

- **OS**: Linux (Ubuntu 20.04/22.04) or Windows with WSL2
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or higher
- **Git**: For cloning the repository
- **Git LFS**: For large file handling (optional)

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/text-to-video-generation.git
cd text-to-video-generation
```

### Step 2: Create Virtual Environment

**Option A: Using venv (recommended)**
```bash
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

**Option B: Using conda**
```bash
conda create -n t2v python=3.10
conda activate t2v
```

### Step 3: Install PyTorch with CUDA Support

**For CUDA 11.8:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

**Verify CUDA installation:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Optional: Install flash-attention (for better performance)**
```bash
pip install flash-attn --no-build-isolation
```

**Optional: Install xformers (memory-efficient attention)**
```bash
pip install xformers
```

### Step 5: Install Model-Specific Dependencies

**For ModelScope:**
```bash
pip install modelscope
```

**For CogVideoX-2B:**
```bash
pip install transformers>=4.35.0 sentencepiece protobuf
```

**For AnimateDiff LoRA:**
```bash
pip install peft diffusers accelerate
```

---

## 📁 Dataset Setup

### Model 1: ModelScope (Something-Something V2)

1. **Download the dataset:**
   - Visit: https://developer.qualcomm.com/software/ai-datasets/something-something
   - Register and download the dataset (academic use only)

2. **Organize the data:**
   ```bash
   mkdir -p data/something-something-v2/
   # Extract videos and annotations
   # Expected structure:
   # data/something-something-v2/
   # ├── train/
   # │   ├── video_0001.mp4
   # │   ├── video_0002.mp4
   # │   └── ...
   # └── annotations.json
   ```

3. **Update config:**
   Edit `configs/modelscope_config.yaml` with correct paths

### Model 2: CogVideoX-2B (UBC Fashion)

1. **Download the dataset:**
   - Visit: https://vision.cs.ubc.ca/datasets/fashion/
   - Download fashion videos (academic use only)

2. **Organize the data:**
   ```bash
   mkdir -p data/ubc-fashion/videos/
   # Extract videos and create captions
   # Expected structure:
   # data/ubc-fashion/
   # ├── videos/
   # │   ├── dress_001.mp4
   # │   ├── dress_002.mp4
   # │   └── ...
   # └── captions.json
   ```

3. **Update config:**
   Edit `configs/cogvideox_config.yaml` with correct paths

### Model 3: AnimateDiff LoRA (Anime)

1. **Prepare anime dataset:**
   - Collect 200 anime video clips (ensure you have rights to use them)
   - Each video should be 2-4 seconds long

2. **Organize the data:**
   ```bash
   mkdir -p data/anime/videos/
   # Add videos and create captions file
   # Expected structure:
   # data/anime/
   # ├── videos/
   # │   ├── anime_001.mp4
   # │   ├── anime_002.mp4
   # │   └── ...
   # └── captions.txt
   ```

3. **Update config:**
   Edit `configs/animatediff_config.yaml` with correct paths

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# GPU settings
CUDA_VISIBLE_DEVICES=0

# Hugging Face token (for downloading models)
HF_TOKEN=your_huggingface_token_here

# Weights & Biases (optional, for logging)
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=text-to-video-generation

# Cache directories
HF_HOME=./cache/huggingface
TORCH_HOME=./cache/torch
```

### Download Pre-trained Models

**CogVideoX-2B:**
```bash
python -c "from diffusers import CogVideoXPipeline; CogVideoXPipeline.from_pretrained('THUDM/CogVideoX-2b')"
```

**AnimateDiff:**
```bash
python -c "from diffusers import MotionAdapter; MotionAdapter.from_pretrained('animatediff-motion-adapter-v2')"
```

---

## 📊 Running the Models

### Model 1: ModelScope

**Open the training notebook:**
```bash
jupyter notebook notebooks/modelscope/model_1_production_code_google_colab.ipynb
```

**Or run training script (if available):**
```bash
python train_modelscope.py --config configs/modelscope_config.yaml
```

### Model 2: CogVideoX-2B

**Open the training notebook:**
```bash
jupyter notebook notebooks/cogvideox/model_2_Fashion_Dataset_final_p1.ipynb
```

**Or run training script (if available):**
```bash
python train_cogvideox.py --config configs/cogvideox_config.yaml
```

### Model 3: AnimateDiff LoRA

**Open the training notebook:**
```bash
jupyter notebook notebooks/animatediff/model_3_Final-Model3.ipynb
```

**Or run training script (if available):**
```bash
python train_animatediff_lora.py --config configs/animatediff_config.yaml
```

---

## 🧪 Inference (After Training)

### Quick Inference Examples

**ModelScope:**
```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("path/to/trained/model", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

video = pipe("A person picking up a cup", num_frames=24).frames[0]
```

**CogVideoX-2B:**
```python
from diffusers import CogVideoXPipeline
import torch

pipe = CogVideoXPipeline.from_pretrained("path/to/fine-tuned/model", torch_dtype=torch.bfloat16)
pipe.enable_vae_slicing()
pipe = pipe.to("cuda")

video = pipe("A red evening dress with lace patterns", num_frames=48).frames[0]
```

**AnimateDiff LoRA:**
```python
from diffusers import AnimateDiffPipeline, MotionAdapter
from peft import PeftModel
import torch

adapter = MotionAdapter.from_pretrained("animatediff-motion-adapter-v2")
pipe = AnimateDiffPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", motion_adapter=adapter)
pipe.unet = PeftModel.from_pretrained(pipe.unet, "path/to/lora/weights")
pipe = pipe.to("cuda")

video = pipe("Anime girl with blue hair walking", num_frames=16).frames[0]
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: CUDA Out of Memory**
```bash
# Solution 1: Reduce batch size in config
batch_size: 2  # or 1

# Solution 2: Enable gradient checkpointing
gradient_checkpointing: true

# Solution 3: Enable mixed precision
mixed_precision: "fp16"  # or "bf16"

# Solution 4: Enable VAE slicing (for CogVideoX)
pipe.enable_vae_slicing()
```

**Issue: Model download fails**
```bash
# Set Hugging Face token
export HF_TOKEN=your_token_here

# Or use environment variable
HF_TOKEN=your_token_here python script.py
```

**Issue: Flash Attention installation fails**
```bash
# Try without build isolation
pip install flash-attn --no-build-isolation

# Or skip flash attention (slightly slower)
# Just don't enable it in config
```

**Issue: xformers not compatible**
```bash
# Install specific version matching PyTorch
pip install xformers==0.0.22.post7
```

### Performance Optimization

**Enable TF32 (for Ampere GPUs):**
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**Use torch.compile() (PyTorch 2.0+):**
```python
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

**Gradient accumulation for memory:**
```yaml
batch_size: 1
gradient_accumulation_steps: 16  # Effective batch size = 16
```

---

## 📚 Additional Resources

- **Project Documentation**: See [docs/](../docs/) folder
- **Model 1 Details**: [docs/MODEL1.md](MODEL1.md)
- **Model 2 Details**: [docs/MODEL2.md](MODEL2.md)
- **Model 3 Details**: [docs/MODEL3.md](MODEL3.md)
- **Comprehensive Report**: [docs/Team5-Workbook2.docx](Team5-Workbook2.docx)

---

## 🎓 Academic Use

This project is part of SJSU DATA 298B Capstone. For academic use:

1. **Cite the datasets:**
   - Something-Something V2
   - UBC Fashion Dataset
   - Custom Anime Dataset (with proper attribution)

2. **Respect licenses:**
   - Academic use only for datasets
   - Follow model licenses (Apache 2.0, CreativeML Open RAIL-M)

3. **Attribution:**
   - Mention this project if you use any code or insights
   - Link to the original repository

---

## 💬 Support

**For issues:**
- Open a GitHub issue
- Contact: [Your Email]

**For collaboration:**
- LinkedIn: [Your LinkedIn]
- Portfolio: [Your Website]

---

**Happy training! 🚀**

*Last Updated: November 2024*
