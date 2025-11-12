#  Multi-Model Text-to-Video Generation System

**SJSU DATA 298B Capstone Project | January 2025**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-A100%20%7C%20H200-76B900.svg)](https://www.nvidia.com/)

> **A production-ready comparative study of three state-of-the-art text-to-video generation architectures, achieving up to 82.2% quality improvement with parameter-efficient training.**

---

##  Key Achievements

| Model | Domain | Training Time | Key Metric | Highlight |
|-------|--------|---------------|------------|-----------|
| **ModelScope** | Human Actions | 9.48 hours | 0.1036 final loss | Scaled to 10K videos |
| **CogVideoX-2B** | Fashion | 62 minutes | **82.2% quality improvement** | 4.5/10 → 8.2/10 rating |
| **AnimateDiff LoRA** | Anime | **64 minutes** | 30.2% temporal consistency | Only 16M params (1% of base) |

###  Notable Results

- ** Fashion Model**: Achieved 82.2% quality improvement with +100% fabric pattern accuracy and +125% sleeve detail precision
- ** Speed Champion**: AnimateDiff trained in just 8 minutes using parameter-efficient LoRA fine-tuning
- ** Scale Master**: ModelScope successfully trained on 10,000 videos from Something-Something V2 dataset
- ** Efficiency Expert**: AnimateDiff uses only 16M trainable parameters (1% of base model) while achieving 30.2% temporal consistency improvement

---

##  Architecture Comparison

### Model 1: ModelScope (Diffusion-based)
```
Text Encoder → UNet 3D Diffusion → Temporal Attention → Video Output
└─ Dataset: Something-Something V2 (10,000 videos)
└─ Focus: Human action recognition and generation
└─ Training: 9.48 hours on A100 80GB
```

**Technical Highlights:**
- 3D UNet with temporal convolutions for motion coherence
- Trained on diverse human action dataset
- Production-ready loss convergence (0.1036)

### Model 2: CogVideoX-2B (Expert Fine-tuning)
```
CLIP Text Encoder → 3D VAE → Expert Transformer → Fashion Video
└─ Dataset: UBC Fashion Dataset (480 videos)
└─ Focus: Fashion garment generation with fine details
└─ Training: 62 minutes on A100 80GB
```

**Technical Highlights:**
- **82.2% quality improvement** (4.5/10 → 8.2/10)
- +100% improvement in fabric pattern rendering
- +100% improvement in dress fit accuracy
- +125% improvement in sleeve detail generation
- Expert fine-tuning on domain-specific dataset

### Model 3: AnimateDiff LoRA (Parameter-Efficient)
```
Stable Diffusion + Motion Module + LoRA Adapters → Anime Video
└─ Dataset: Custom Anime Dataset (200 videos)
└─ Focus: Anime-style animation generation
└─ Training: 8 minutes on H200 140GB
```

**Technical Highlights:**
- **Parameter-efficient**: Only 16M trainable params (1% of base model)
- **30.2% temporal consistency improvement**
- LoRA rank-16 adaptation for motion coherence
- Ultra-fast training (8 minutes)

---

##  Repository Structure

```
text-to-video-generation/
├── README.md                          # This file
├── models/                            # Model architectures and configs
│   ├── modelscope/                    # ModelScope implementation details
│   ├── cogvideox/                     # CogVideoX-2B architecture
│   └── animatediff/                   # AnimateDiff LoRA setup
├── notebooks/                         # Training and evaluation notebooks
│   ├── modelscope/
│   │   ├── model_1_production_code_google_colab.ipynb
│   │   └── model_1_Video_Evaluation_Metrics.ipynb
│   ├── cogvideox/
│   │   └── model_2_Fashion_Dataset_final_p1.ipynb
│   └── animatediff/
│       └── model_3_Final-Model3.ipynb
├── results/                           # Generated videos and metrics
│   ├── modelscope/                    # Model 1 outputs
│   ├── cogvideox/                     # Model 2 outputs (fashion)
│   └── animatediff/                   # Model 3 outputs (anime)
├── docs/                              # Documentation
│   ├── Team5-Workbook2.docx          # Comprehensive project report
│   ├── MODEL1.md                      # ModelScope documentation
│   ├── MODEL2.md                      # CogVideoX documentation
│   └── MODEL3.md                      # AnimateDiff documentation
└── configs/                           # Configuration files
    ├── modelscope_config.yaml
    ├── cogvideox_config.yaml
    └── animatediff_config.yaml
```

---

##  Project Objectives

This capstone project demonstrates:

1. **Multi-Architecture Expertise**: Comparative implementation of three distinct text-to-video generation paradigms
2. **Scalability**: From 200 videos (AnimateDiff) to 10,000 videos (ModelScope)
3. **Efficiency**: Parameter-efficient training achieving 30.2% improvement with 1% trainable parameters
4. **Domain Adaptation**: Successful fine-tuning across diverse domains (actions, fashion, anime)
5. **Production Readiness**: Complete evaluation pipelines with quantitative metrics

---

## 🔬 Evaluation Metrics

### Video Quality Assessment
- **Inception Score (IS)**: Measures diversity and quality
- **Fréchet Video Distance (FVD)**: Compares generated vs. real video distributions
- **CLIP Score**: Text-video alignment accuracy
- **Temporal Consistency**: Frame-to-frame coherence (SSIM, PSNR)

### Model-Specific Results

**ModelScope (Human Actions)**
- Final Training Loss: 0.1036
- Dataset Scale: 10,000 videos
- Convergence: Stable after 9.48 hours

**CogVideoX-2B (Fashion)**
- Overall Quality: **82.2% improvement** (4.5/10 → 8.2/10)
- Fabric Patterns: +100% accuracy
- Dress Fit: +100% accuracy
- Sleeve Details: +125% accuracy
- Training Efficiency: 62 minutes

**AnimateDiff LoRA (Anime)**
- Temporal Consistency: +30.2% improvement
- Training Parameters: 16M (1% of base model)
- Training Time: 8 minutes (fastest)
- Motion Coherence: Significantly improved over base model

---

##  Technical Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | PyTorch 2.0+, Diffusers, Transformers |
| **Models** | ModelScope, CogVideoX-2B, AnimateDiff + LoRA |
| **Hardware** | NVIDIA A100 (80GB), H200 (140GB) |
| **Training** | Mixed Precision (FP16/BF16), Gradient Checkpointing |
| **Datasets** | Something-Something V2, UBC Fashion, Custom Anime |
| **Evaluation** | CLIP, FVD, IS, SSIM, PSNR |

---

##  Getting Started

### Prerequisites
```bash
Python 3.10+
CUDA 11.8+
16GB+ GPU RAM (inference)
80GB+ GPU RAM (training)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/text-to-video-generation.git
cd text-to-video-generation
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up datasets** (see individual model documentation in [docs/](docs/))

### Quick Start - Inference

**ModelScope (Human Actions)**
```python
# See notebooks/modelscope/model_1_production_code_google_colab.ipynb
# Generates videos of human actions from text prompts
```

**CogVideoX-2B (Fashion)**
```python
# See notebooks/cogvideox/model_2_Fashion_Dataset_final_p1.ipynb
# Generates fashion garment videos with 82.2% quality improvement
```

**AnimateDiff LoRA (Anime)**
```python
# See notebooks/animatediff/model_3_Final-Model3.ipynb
# Fast anime video generation in 8 minutes
```

---

##  Training Details

### Hardware Requirements

| Model | GPU | VRAM | Training Time |
|-------|-----|------|---------------|
| ModelScope | A100 | 80GB | 9.48 hours |
| CogVideoX-2B | A100 | 80GB | 62 minutes |
| AnimateDiff LoRA | H200 | 140GB | 8 minutes |

### Dataset Information

| Model | Dataset | Size | Domain |
|-------|---------|------|--------|
| ModelScope | Something-Something V2 | 10,000 videos | Human actions |
| CogVideoX-2B | UBC Fashion | 480 videos | Fashion garments |
| AnimateDiff | Custom Anime | 200 videos | Anime animation |

---

##  Detailed Results

### Model 1: ModelScope
- **Training Loss Progression**: Converged to 0.1036
- **Dataset Scale**: Successfully scaled to 10,000 videos
- **Motion Coherence**: Stable temporal consistency across frames
- **Action Recognition**: High fidelity human action generation

### Model 2: CogVideoX-2B
- **Overall Quality Score**: 4.5/10 → 8.2/10 (**82.2% improvement**)
- **Fabric Pattern Accuracy**: +100% improvement
- **Dress Fit Realism**: +100% improvement
- **Sleeve Detail Rendering**: +125% improvement
- **Training Efficiency**: 62 minutes on 480 videos

### Model 3: AnimateDiff LoRA
- **Temporal Consistency**: +30.2% improvement over base model
- **Parameter Efficiency**: Only 16M trainable params (1% of base)
- **Training Speed**: 8 minutes (fastest among all models)
- **Motion Quality**: Smooth anime-style animations
- **Memory Efficiency**: LoRA enables low-resource fine-tuning

---

##  Academic Context

**Course**: DATA 298B - Master's Capstone Project
**Institution**: San José State University
**Semester**: Fall 2024
**Evaluation Focus**:
- Research rigor and methodology
- Technical implementation quality
- Comparative analysis depth
- Production readiness
- Documentation completeness

---

##  Use Cases

### ModelScope - Human Action Generation
- **Robotics**: Training data for action recognition systems
- **Gaming**: Procedural NPC animation generation
- **Film/VFX**: Rapid prototyping of action sequences

### CogVideoX-2B - Fashion Video Generation
- **E-commerce**: Product video generation for online stores
- **Fashion Design**: Virtual garment visualization
- **Marketing**: Automated fashion content creation

### AnimateDiff LoRA - Anime Production
- **Animation Studios**: Rapid scene prototyping
- **Content Creation**: Social media animation generation
- **Game Development**: Cinematic cutscene creation

---

##  Key Insights & Learnings

### 1. **Scale vs. Efficiency Trade-off**
   - ModelScope shows that scale (10K videos) requires significant compute (9.48 hours)
   - AnimateDiff demonstrates parameter-efficient learning achieves strong results in 8 minutes
   - Sweet spot: Domain-specific fine-tuning (CogVideoX) balances quality and efficiency

### 2. **Domain Adaptation Matters**
   - Fashion domain achieved 82.2% improvement through expert fine-tuning
   - Specialized datasets (480 fashion videos) outperform generic large-scale training
   - Quality > Quantity when domain knowledge is embedded

### 3. **Parameter Efficiency is Key**
   - LoRA adaptation (1% trainable params) achieves 30.2% improvement
   - Enables democratization of video generation (lower compute requirements)
   - Production-friendly: faster iteration cycles

### 4. **Architecture Selection Guide**
   | Need | Choose | Reason |
   |------|--------|--------|
   | General-purpose | ModelScope | Broad action coverage |
   | Domain-specific quality | CogVideoX | Expert fine-tuning |
   | Fast iteration / Low resources | AnimateDiff LoRA | Parameter efficiency |

---

##  Documentation

- **[Model 1: ModelScope](docs/MODEL1.md)** - Human action generation architecture and training details
- **[Model 2: CogVideoX-2B](docs/MODEL2.md)** - Fashion video generation with 82.2% improvement
- **[Model 3: AnimateDiff LoRA](docs/MODEL3.md)** - Parameter-efficient anime generation
- **[Comprehensive Report](docs/Team5-Workbook2.docx)** - Full technical documentation with diagrams

---

##  Contributing

This is an academic capstone project. For questions or collaboration:
- **Contact**: [Your Email]
- **LinkedIn**: [Your LinkedIn]
- **Portfolio**: [Your Portfolio Website]

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Important**: Dataset licenses are more restrictive than the code license. See [CITATIONS.md](CITATIONS.md) for complete attribution and licensing information.

---

##  Acknowledgments

### Academic & Research
- **SJSU DATA 298B** - Course and academic guidance

### Datasets
- **Something-Something V2** - Goyal, R., et al. (2017). The "Something Something" Video Database for Learning and Evaluating Visual Common Sense. *ICCV 2017*. [Dataset Link](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
- **Fashion-Text2Video** - Jiang, Y., et al. (2023). Text2Performer: Text-Driven Human Video Generation. *ICCV 2023*. [Dataset Link](https://github.com/yumingj/Fashion-Text2Video)
- **MSR-VTT** - Xu, J., et al. (2016). MSR-VTT: A Large Video Description Dataset for Bridging Video and Language. *CVPR 2016*. [Dataset Link](https://huggingface.co/datasets/friedrichor/MSR-VTT)

### Models & Infrastructure
- **ModelScope** - Alibaba DAMO Academy
- **CogVideoX** - Tsinghua University (THUDM)
- **AnimateDiff** - Anonymous authors
- **Stable Diffusion** - Stability AI & CompVis
- **Hugging Face** - Model architectures, Diffusers, and Transformers libraries
- **NVIDIA** - GPU compute resources (A100 80GB, H200 140GB)

**For complete citations and BibTeX entries**, see [CITATIONS.md](CITATIONS.md)


** If this project interests you, please star the repository!**

*Last Updated: 11th November 2025*
