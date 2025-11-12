# Quick Start Guide

Get up and running with text-to-video generation in minutes!

---

##  5-Minute Setup (Inference Only)

If you just want to try generating videos with the pre-trained models:

### Step 1: Install Dependencies (2 minutes)

```bash
# Clone repo
git clone https://github.com/yourusername/text-to-video-generation.git
cd text-to-video-generation

# Create environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# Install PyTorch + dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate peft
```

### Step 2: Download Pre-trained Model (2 minutes)

```bash
# For CogVideoX (Fashion - Best results)
python -c "from diffusers import CogVideoXPipeline; CogVideoXPipeline.from_pretrained('THUDM/CogVideoX-2b', torch_dtype='float16')"
```

### Step 3: Generate Your First Video (1 minute)

```python
# generate_video.py
from diffusers import CogVideoXPipeline
import torch

# Load model
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

# Generate video
prompt = "A model wearing a red evening dress with flowing sleeves"
video_frames = pipe(
    prompt=prompt,
    num_frames=48,
    guidance_scale=7.5
).frames[0]

# Save video
from PIL import Image
video_frames[0].save("output.gif", save_all=True, append_images=video_frames[1:], duration=125, loop=0)
print("Video saved as output.gif!")
```

Run it:
```bash
python generate_video.py
```

**That's it! You've generated your first video! **

---

##  Model Comparison - Which One to Use?

| Use Case | Model | Why? | Time to Results |
|----------|-------|------|-----------------|
| **Fashion/E-commerce** | CogVideoX-2B | 82.2% quality improvement, best for clothing | 5 minutes (inference) |
| **Anime/Animation** | AnimateDiff LoRA | Fast, efficient, best for anime style | 3 minutes (inference) |
| **Human Actions** | ModelScope | Best for action recognition, general purpose | 10 minutes (inference) |

---

##  Model-Specific Quick Starts

### Model 1: ModelScope (Human Actions)

**Best for**: Robotics, action recognition, general human activities

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "ali-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

video = pipe(
    "A person picking up a cup from the table",
    num_inference_steps=50,
    num_frames=24
).frames[0]

# Save video
from moviepy.editor import ImageSequenceClip
clip = ImageSequenceClip([frame for frame in video], fps=8)
clip.write_videofile("action_video.mp4")
```

**Training**: See [notebooks/modelscope/model_1_production_code_google_colab.ipynb](../notebooks/modelscope/model_1_production_code_google_colab.ipynb)

---

### Model 2: CogVideoX-2B (Fashion)  Recommended

**Best for**: Fashion, e-commerce, garment visualization
**Achievement**: 82.2% quality improvement!

```python
from diffusers import CogVideoXPipeline
import torch

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

video = pipe(
    "A flowing red evening dress with intricate lace patterns",
    num_frames=48,
    guidance_scale=7.5
).frames[0]

# Save video
from moviepy.editor import ImageSequenceClip
clip = ImageSequenceClip(video, fps=8)
clip.write_videofile("fashion_video.mp4")
```

**Fine-tuning**: See [notebooks/cogvideox/model_2_Fashion_Dataset_final_p1.ipynb](../notebooks/cogvideox/model_2_Fashion_Dataset_final_p1.ipynb)

---

### Model 3: AnimateDiff LoRA (Anime)  Fastest

**Best for**: Anime, animation studios, content creation
**Achievement**: 8-minute training, 30.2% temporal improvement!

```python
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
import torch

# Load motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")

# Load pipeline
pipe = AnimateDiffPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    motion_adapter=adapter,
    torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Generate anime video
video = pipe(
    "Anime girl with blue hair walking in cherry blossom garden",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25
).frames[0]

# Save video
from moviepy.editor import ImageSequenceClip
clip = ImageSequenceClip(video, fps=8)
clip.write_videofile("anime_video.mp4")
```

**LoRA Training**: See [notebooks/animatediff/model_3_Final-Model3.ipynb](../notebooks/animatediff/model_3_Final-Model3.ipynb)

---

##  Pro Tips

### Memory-Efficient Generation

If you get CUDA out of memory errors:

```python
# Enable memory-efficient features
pipe.enable_model_cpu_offload()  # Move modules to CPU when not in use
pipe.enable_vae_slicing()  # Process video in slices
pipe.enable_xformers_memory_efficient_attention()  # If xformers installed

# Reduce batch size or frames
video = pipe(prompt, num_frames=16)  # Instead of 48
```

### Faster Generation

```python
# Reduce inference steps
video = pipe(prompt, num_inference_steps=25)  # Instead of 50

# Use DDIM scheduler (faster)
from diffusers import DDIMScheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
```

### Better Quality

```python
# Increase guidance scale
video = pipe(prompt, guidance_scale=9.0)  # Higher = more prompt adherence

# More inference steps
video = pipe(prompt, num_inference_steps=50)  # More steps = better quality
```

---

##  Next Steps

### For Users (Just Generate Videos)
1.  You're done! Generate videos with the quick starts above
2. Experiment with different prompts
3. Try different models for different use cases

### For Researchers (Train Your Own)
1. **Setup environment**: Follow [docs/SETUP.md](SETUP.md)
2. **Prepare datasets**: Download and organize data
3. **Choose a model**:
   - ModelScope: Large-scale training (9.48 hours, A100)
   - CogVideoX: Domain expert fine-tuning (62 minutes, A100)
   - AnimateDiff LoRA: Parameter-efficient (8 minutes, H200)
4. **Run training**: Open respective Jupyter notebooks
5. **Evaluate**: Use evaluation notebooks

### For Recruiters (Understand the Project)
1. **Read the main README**: [README.md](../README.md) - Key achievements
2. **Review model docs**:
   - [Model 1 (ModelScope)](MODEL1.md) - Scale: 10K videos
   - [Model 2 (CogVideoX)](MODEL2.md) - Quality: 82.2% improvement
   - [Model 3 (AnimateDiff)](MODEL3.md) - Efficiency: 8 minutes training
3. **Check comprehensive report**: [Team5-Workbook2.docx](Team5-Workbook2.docx)

---

##  Common Issues & Solutions

### Issue: "CUDA out of memory"
**Solution**: Use memory-efficient features (see Pro Tips above)

### Issue: "Model download fails"
**Solution**:
```bash
export HF_TOKEN=your_huggingface_token
# Or use huggingface-cli login
pip install huggingface_hub
huggingface-cli login
```

### Issue: "Slow generation"
**Solution**: Reduce inference steps and use DDIM scheduler (see Pro Tips)

### Issue: "Low quality videos"
**Solution**: Increase guidance_scale and num_inference_steps

---

##  Benchmarks (For Reference)

| Model | Inference Time | GPU Memory | Video Quality | Best Use Case |
|-------|----------------|------------|---------------|---------------|
| ModelScope | ~30s (24 frames) | ~20GB | Good | Human actions |
| CogVideoX-2B | ~45s (48 frames) | ~30GB | **Excellent** | Fashion |
| AnimateDiff LoRA | ~15s (16 frames) | ~12GB | Very Good | Anime |

*Times on A100 40GB with optimizations enabled*

---

##  Sample Prompts

### Fashion (CogVideoX)
- "A model wearing a flowing blue evening gown with sequins"
- "Red leather jacket with white t-shirt, casual style"
- "Black formal business suit with tie, professional setting"
- "Summer floral dress with short sleeves in garden"

### Anime (AnimateDiff)
- "Anime girl with long blue hair walking in cherry blossoms"
- "Anime character running through cyberpunk city at night"
- "Magical girl transformation sequence with sparkles"
- "Samurai warrior standing in bamboo forest, moonlight"

### Human Actions (ModelScope)
- "A person picking up a cup from the table"
- "Someone opening a door and walking through"
- "A person waving goodbye with hand"
- "Pushing a box from left to right"

---

##  Community & Support

- **Issues**: Open a GitHub issue
- **Questions**: [Your Email]
- **Discussions**: GitHub Discussions (if enabled)
- **LinkedIn**: [Your LinkedIn Profile]

---

##  Like This Project?

If you find this project helpful:
1.  Star the repository
2.  Share on LinkedIn/Twitter
3.  Write about your experience
4.  Contribute improvements

---

**Now go create something amazing! **

*Last Updated: November 2024*
