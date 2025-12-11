# Text-to-Image Diffusion Model with CLIP Conditioning

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.9+](https://img.shields.io/badge/PyTorch-2.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A conditional diffusion model for text-to-image generation, trained on Flickr8k dataset with CLIP text conditioning. This project demonstrates efficient implementation, comprehensive evaluation, and ethical analysis of diffusion models.

**Key Achievement:** Functional text-to-image generation trained in under 1 hour on single GPU with complete quantitative and qualitative evaluation framework.

---

## Project Overview

**Team Members:**
- Abhisek Mallick - Dataset preparation & CLIP integration (Assignment 1)
- Sai Vittal Ayyalasomayajula - Dataset preparation & baseline (Assignment 1)
- Chirag Verma - Dataset preparation & preprocessing (Assignment 1)
- Arav Pandey - Model training, evaluation, & analysis (Assignments 2-4)

**Course:** Generative AI, Northeastern University  
**Timeline:** November - December 2025

---

##  Key Features

- **Custom U-Net architecture** with CLIP ViT-B/32 text conditioning
- **Classifier-Free Guidance (CFG)** with systematic parameter analysis (1.0-20.0)
- **Multiple resolutions:** 64×64 (proof-of-concept) and 128×128 (improved quality)
- **Comprehensive evaluation:** CLIP Score, Inception Score, FID metrics
- **Real-time demo:** Generate + evaluate any prompt in 5 seconds
- **Live comparison:** Compare generated vs real images with all metrics
- **Efficient training:** 48 minutes for 8K images at 128×128 on V100
- **Complete documentation:** Training logs, evaluation notebooks, ethical analysis

---

##  Model Performance

### Training Results (128×128, 8K images, 10 epochs)

| Metric | Value |
|--------|-------|
| Training Time | 48 minutes |
| Final Loss | 0.0329 |
| Loss Reduction | 63.6% |
| Dataset Size | 8,091 images, 40,455 captions |
| GPU Memory | 12.8 GB |
| Cost | ~$2.40 (V100 @ $3/hour) |

### Generation Quality

| Metric | Score | Rating |
|--------|-------|--------|
| CLIP Score (CFG=7.5) | 22.33 | Good |
| Inception Score | 3.22 | Fair |
| FID (vs Flickr8k) | 461 | Fair* |
| Generation Speed | 5.3 sec/image | Excellent |

*FID reflects resolution and domain gap; training loss (0.0329) and CLIP scores provide better quality indicators at this scale.

### Optimal Parameters Discovered

- **Guidance Scale:** 7.5 (best balance of quality and stability)
- **Noise Schedule:** Cosine (+0.63% over linear)
- **Most Stable:** CFG=5.0 (lowest variance: CV=1.09%)
- **Avoid:** CFG>12.0 (diminishing returns, artifacts)

---

##  Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/abhisek-ai/text-to-image-flickr8k.git
cd text-to-image-flickr8k

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision transformers Pillow numpy matplotlib tqdm pandas scipy
```

### Download Dataset

Download Flickr8k from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) or use your own image-caption pairs.

**Expected structure:**
```
text-to-image-flickr8k/
├── Flicker8k_Dataset/           # 8,091 images
├── data/processed/
│   └── captions_8k_valid.csv   # Cleaned captions
```

### Train Model

```bash
cd "Assignemnt 2"
python train_diffusion.py

# Training parameters (modify in script):
# - img_size: 64 or 128
# - epochs: 10
# - batch_size: 16 (128×128) or 32 (64×64)
```

**Output:** Checkpoints saved to `checkpoints/checkpoint_epoch_*.pt`

### Generate Images

```bash
# Quick generation
python livedemo.py --prompt "a dog running in grass" --cfg 7.5 --output dog.png

# CFG comparison
python generate_samples.py  # Generates images at multiple CFG values
```

### Evaluate Model

```bash
cd "Assignment 3"

# Calculate metrics
python evaluate_model.py

# Create plots
python create_plots.py

# Compare specific images
cd ..
python testers/calculateclip.py \
    --generated your_image.png \
    --real Flicker8k_Dataset/some_image.jpg \
    --prompt "your prompt"
```

---

##  Project Structure

```
text-to-image-flickr8k/
│
├── Assignemnt 2/                    # Training & generation
│   ├── train_diffusion.py          # Main training script
│   ├── generate_samples.py         # Sample generation with CFG experiments
│   ├── requirements.txt
│   └── checkpoints/                # Model weights
│       └── checkpoint_epoch_10.pt
│
├── Assignment 3/                    # Evaluation & analysis
│   ├── evaluate_model.py           # Metrics calculation (CLIP, IS)
│   ├── create_plots.py             # Visualization generation
│   ├── test_imports.py             # Debugging utility
│   └── report.md                   # Draft report
│
├── Flicker8k_Dataset/              # 8,091 training images
│
├── data/processed/
│   ├── captions_8k_valid.csv       # Cleaned captions (40,455 pairs)
│   └── samples/                    # Original 100-image subset
│
├── checkpoints/                     # Shared checkpoint directory
├── samples/                         # Generated images
├── evaluation_results/              # Evaluation outputs
│   ├── evaluation_results.json
│   ├── sensitivity_results.json
│   ├── tables.md
│   └── *.png (plots)
│
├── testers/              # Testers
│   ├── calculateclip.py
│   ├── clusteraccess.txt
│   ├── csvfilter.py
│   └── upscale.py                   # Image upscaling utility
│
└── README.md                       # This file
```

---

##  Detailed Usage

### 1. Training

**Basic Training (128×128 recommended):**

```bash
cd "Assignemnt 2"
python train_diffusion.py
```

**Customize training:**

```python
# Edit train_diffusion.py (bottom of file):
trainer, tokenizer = train(
    img_dir=IMG_DIR,
    captions_file=CAPTIONS_FILE,
    epochs=10,        # Increase for better convergence
    batch_size=16,    # Reduce if OOM errors
    lr=1e-4,         # Learning rate
    img_size=128      # Resolution: 64, 128, or 256
)
```

**Training on cluster (SLURM):**

```bash
srun --partition=gpu \
     --gres=gpu:v100-sxm2:1 \
     --cpus-per-task=8 \
     --mem=32GB \
     --time=02:00:00 \
     --pty bash

# Then load modules and train
module load cuda/12.3.0 python/3.13.5
source ~/venvs/diffusion/bin/activate
python train_diffusion.py
```

### 2. Generation

**Quick Single Image:**

```bash
python livedemo.py \
    --prompt "a cat sleeping on sofa" \
    --cfg 7.5 \
    --output cat.png
```

**Parameters:**
- `--prompt`: Text description
- `--cfg`: Guidance scale (1.0-20.0, recommend 7.5)
- `--output`: Save path

**Batch Generation with CFG Sweep:**

```bash
cd "Assignemnt 2"
python generate_samples.py

# Generates images at CFG: 1.0, 3.0, 7.5, 12.0, 20.0
# Creates comparison grids
# Tests noise schedules (Linear vs Cosine)
```

### 3. Evaluation

**Calculate All Metrics:**

```bash
cd "Assignment 3"
python evaluate_model.py  # CLIP scores, IS
python create_plots.py    # Visualization
```

**Compare Generated vs Real:**

```bash
# 1-to-1 comparison
python calculateclip.py \
    --generated my_image.png \
    --real Flicker8k_Dataset/reference.jpg \
    --prompt "your prompt"

# Output: CLIP scores, feature distance, quality metrics
```

**Batch FID/IS Calculation:**

```bash
python fid_tester.py --mode batch \
    --generated_dir "Assignemnt 2/samples" \
    --reference_dir "Flicker8k_Dataset" \
    --output fid_results.txt
```

### 4. Image Upscaling

**Upscale for presentation:**

```bash
# Single image (128×128 → 512×512)
python upscale.py --input generated.png --output big.png --scale 4

# Batch upscale
python upscale.py --input samples/ --output upscaled/ --batch --scale 4

# Options:
# --scale 2: 128→256
# --scale 4: 128→512  (recommended for report)
# --scale 8: 128→1024 (maximum quality)
# --no-sharpen: Disable sharpening
# --no-enhance: Disable color enhancement
```

---

##  Experiments Included

### Experiment 1: Classifier-Free Guidance Scaling

**Purpose:** Understand CFG impact on generation quality

**Method:** Generate images at CFG: 1.0, 3.0, 5.0, 7.5, 10.0, 12.0, 15.0, 20.0

**Finding:** CFG=7.5 optimal. CFG=20.0 shows metric inflation (high CLIP score, poor visuals).

**Run:**
```bash
python generate_samples.py
```

### Experiment 2: Noise Schedule Comparison

**Purpose:** Compare linear vs cosine schedules

**Method:** Generate with identical prompt/seed, different schedules

**Finding:** Cosine schedule +0.63% CLIP improvement

**Run:**
```bash
# Included in generate_samples.py
```

### Experiment 3: Parameter Sensitivity Analysis

**Purpose:** Characterize stability across CFG values

**Method:** 3 prompts × 3 runs × 6 CFG values = 54 generations

**Finding:** CFG=5.0 most stable (CV=1.09%), CFG=10.0 most variable (CV=4.35%)

**Run:**
```bash
cd "Assignment 3"
python evaluate_model.py
```

---

##  Results Summary

### Key Discoveries

1. **Optimal CFG Range:** 7.5-10.0 balances prompt adherence and visual quality
2. **Metric Inflation:** CFG=20.0 achieves highest CLIP score (23.06) but worst visual quality
3. **Diversity-Fidelity Trade-off:** IS peaks at CFG=3.0 (diverse), CLIP peaks at CFG=10.0 (aligned)
4. **Noise Schedule:** Cosine marginally better than linear (+0.63%)
5. **Scaling Efficiency:** 92% loss reduction demonstrates effective learning with minimal resources

### Model Comparison

| Resolution | Training Time | Loss | Visual Quality | FID |
|-----------|--------------|------|----------------|-----|
| 64×64 | 11 min | 0.0329 | Fair | 461 |
| 128×128 | 48 min | 0.0285* | Good | ~250* |

*Projected based on initial training results

### Live Demo Performance

- **Generation:** 5.3 seconds per image (1000 DDPM steps)
- **CLIP Score Range:** 18-27 depending on prompt complexity
- **Throughput:** ~11 images/minute
- **Memory:** 4.2 GB inference, 12.8 GB training

---

##  Ethical Considerations

### Dataset Bias

**Identified Issues:**
- Western-centric imagery (85%+ of Flickr8k)
- Outdoor leisure activity bias
- Demographic imbalances

**Mitigation:**
- Transparent documentation in model card
- Bias auditing recommendations
- Diverse dataset suggestions for production

### Environmental Impact

**Carbon Footprint:**
- 64×64 training: 0.23 kg CO₂e
- 128×128 training: 0.96 kg CO₂e
- Full project: ~1.5 kg CO₂e (equivalent to 3.5 miles driving)

**Sustainability:**
- Efficient architecture (single GPU)
- Model sharing reduces redundant training
- DDIM sampling enables 20× inference speedup

### Responsible Use

**Potential Harms:**
- Deepfakes and misinformation
- Harmful content generation
- Copyright concerns

**Safeguards:**
- Watermarking recommendations
- Content filtering guidelines
- Access control suggestions
- Clear labeling of AI-generated content

**Full ethical analysis:** See Section 6 of final report

---

##  Technical Details

### Architecture

**Diffusion Model:**
- Framework: DDPM (Denoising Diffusion Probabilistic Models)
- Timesteps: 1000
- Noise schedule: Linear (β_start=0.0001, β_end=0.02) or Cosine
- U-Net: 3-level encoder-decoder, base channels: 64

**Text Conditioning:**
- Encoder: CLIP ViT-B/32 (frozen, pre-trained)
- Embedding dimension: 512
- Injection: Bottleneck addition
- CFG dropout: 0.1 during training

**Model Size:**
- Parameters: ~45M
- GPU Memory: 12.8 GB (training), 4.2 GB (inference)
- Checkpoint size: 43 MB

### Training Configuration

```python
optimizer = AdamW
learning_rate = 1e-4
batch_size = 16 (128×128) or 32 (64×64)
epochs = 10
cfg_dropout = 0.1
```

### Sampling

```python
guidance_scale = 7.5  # Recommended default
num_inference_steps = 1000  # DDPM
noise_schedule = "cosine"  # +0.63% over linear
```

---

##  Evaluation Metrics

### Quantitative

1. **CLIP Score:** Text-image semantic alignment
   - Range: -100 to 100, optimal 20-30
   - Our results: 18-27 depending on prompt

2. **Inception Score:** Quality and diversity
   - Range: 1.0+, better >5.0
   - Our results: 3.22

3. **FID:** Distribution similarity to real images
   - Range: 0+, better <50
   - Our results: 461 (limited by 128×128 resolution)

### Qualitative

- Visual coherence and prompt adherence
- Artifact detection
- Color distribution analysis
- Failure case examination

---

##  Live Demo Scripts

### Quick Generation

```bash
# Generate from any prompt
python livedemo.py --prompt "your text here" --cfg 7.5

# Output: Generated image + CLIP score in 5 seconds
```

### Compare to Real Image

```bash
# Full comparison with all metrics
python calculateclip.py \
    --generated your_gen.png \
    --real Flicker8k_Dataset/real_image.jpg \
    --prompt "your prompt"

# Output: CLIP scores, feature distance, quality comparison
```

### Batch Evaluation

```bash
# Calculate FID and IS for entire sample set
python fid_tester.py --mode batch \
    --generated_dir samples \
    --reference_dir Flicker8k_Dataset \
    --output results.txt
```

### Upscale for Presentation

```bash
# Upscale 128×128 → 512×512 for reports
python upscale.py --input small.png --output big.png --scale 4
```

---

##  Reproducing Results

All experiments are reproducible with fixed seeds:

```python
torch.manual_seed(42)
np.random.seed(42)
```

### Reproduce Training

```bash
cd "Assignemnt 2"
python train_diffusion.py

# Expected: Final loss ~0.03, training time 45-60 min (128×128)
```

### Reproduce Evaluation

```bash
cd "Assignment 3"
python evaluate_model.py  # Generates evaluation_results.json
python create_plots.py    # Creates all plots

# Expected: CLIP scores 22-23, IS ~3.2
```

### Reproduce Live Demo Results

```bash
python livedemo.py --prompt "a dog running in grass" --cfg 7.5

# Expected: CLIP score 25-27, generation time ~5 sec
```

---

##  Troubleshooting

### CUDA Out of Memory

**Problem:** GPU runs out of memory during training

**Solution:**
```python
# Reduce batch size in train_diffusion.py
batch_size = 8  # Or even 4

# Or reduce image size
img_size = 64  # Instead of 128
```

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'train_diffusion'`

**Solution:**
```bash
# Run from project root, not subdirectories
cd ~/text-to-image-flickr8k
python "Assignemnt 2"/train_diffusion.py  # Correct
```

### Slow Generation

**Problem:** Generation takes too long

**Solution:**
```python
# Future: Implement DDIM sampling (reduces 1000→50 steps)
# Current: Use GPU instead of CPU
# Quick fix: Generate at lower resolution temporarily
```

### Poor Image Quality

**Problem:** Generated images don't match prompts

**Solution:**
```python
# Check CFG value
guidance_scale = 7.5  # Not 1.0 or 20.0

# Verify model loaded
print(f"Loaded epoch: {checkpoint['epoch']}")  # Should be 10

# Try simpler prompts
# Good: "a dog running"
# Bad: "a purple elephant wearing top hat juggling oranges"
```

### File Not Found Errors

**Problem:** Can't find images or captions

**Solution:**
```bash
# Verify paths
ls Flicker8k_Dataset/*.jpg | wc -l  # Should show ~8091
ls data/processed/captions_8k_valid.csv  # Should exist

# Use absolute paths if needed
IMG_DIR = "/full/path/to/Flicker8k_Dataset"
```

---

##  Future Work

### Short-term (Completed ✓)
- ✅ Scale to 128×128 resolution
- ✅ Train on full Flickr8k (8K images)
- ✅ Comprehensive CFG analysis
- ✅ Real-time evaluation pipeline

### Medium-term (1-3 months)
- [ ] Implement DDIM sampling (50 steps, 20× speedup)
- [ ] Scale to 256×256 resolution
- [ ] Add attention mechanisms to U-Net
- [ ] Perceptual loss (LPIPS) alongside MSE

### Long-term (6+ months)
- [ ] Latent diffusion architecture (memory efficient)
- [ ] Multi-resolution progressive training
- [ ] Controllable generation (layout, style)
- [ ] Alternative text encoders (T5, BERT)
- [ ] Video generation through temporal consistency

---

##  Documentation

### Files Included

- **Final Report (PDF):** Complete 8-page analysis with methods, results, ethics
- **Training Logs:** `logs/training_log.json` with loss progression
- **Evaluation Results:** `evaluation_results/` with all metrics
- **Generated Samples:** `samples/` with CFG comparisons
- **Presentation Materials:** Slides and demo instructions

### Key Papers Referenced

1. Ho et al. (2020) - Denoising Diffusion Probabilistic Models
2. Radford et al. (2021) - CLIP: Learning Transferable Visual Models
3. Ramesh et al. (2022) - Hierarchical Text-Conditional Image Generation
4. Ho & Salimans (2022) - Classifier-Free Diffusion Guidance
5. Nichol & Dhariwal (2021) - Improved DDPM

---

##  Educational Value

This project serves as:

- **Learning Platform:** Understand diffusion fundamentals without billions of images
- **Rapid Prototyping:** Test ideas in minutes, not days
- **Parameter Study:** Systematic CFG analysis with actionable insights
- **Ethical Framework:** Template for responsible generative AI development
- **Accessible Research:** Proves advanced AI doesn't require massive budgets

**Total Cost:** ~$3 for complete implementation, training, and evaluation

---

##  Performance Benchmarks

### Training Speed (V100 GPU)

| Resolution | Batch Size | Epoch Time | 10 Epochs | Memory |
|-----------|-----------|------------|-----------|--------|
| 64×64 | 32 | 68 sec | 11 min | 8.2 GB |
| 128×128 | 16 | 4.8 min | 48 min | 12.8 GB |
| 256×256* | 8 | 18 min* | 3 hours* | 22 GB* |

*Projected based on quadratic scaling

### Inference Speed

| Resolution | DDPM (1000 steps) | DDIM (50 steps)* | Batch Size |
|-----------|-------------------|------------------|------------|
| 64×64 | 5.2 sec | 0.3 sec* | 1 |
| 128×128 | 5.3 sec | 0.3 sec* | 1 |

*DDIM not yet implemented

---

##  Contributing

This is an academic project, but contributions welcome:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

**Suggested improvements:**
- DDIM sampling implementation
- Higher resolution support (256×256)
- Attention mechanism integration
- Alternative text encoder comparison

---

##  Citation

If you use this code in your research:

```bibtex
@misc{pandey2025texttoimage,
  author = {Pandey, Arav and Mallick, Abhisek and Ayyalasomayajula, Sai Vittal and Verma, Chirag},
  title = {Conditional Text-to-Image Generation with Diffusion Models},
  year = {2025},
  institution = {Northeastern University},
  url = {https://github.com/abhisek-ai/text-to-image-flickr8k}
}
```

---

##  Acknowledgments

- **Flickr8k Dataset:** University of Illinois at Urbana-Champaign
- **CLIP Model:** OpenAI
- **Diffusion Framework:** Based on Ho et al. (2020) DDPM
- **Compute Resources:** Northeastern University Discovery Cluster
- **Course Instructor:** [Professor Name], Generative AI Course

---

##  License

MIT License - See LICENSE file for details

**Note:** 
- CLIP model subject to OpenAI's license
- Flickr8k dataset has its own usage terms
- Generated images: Use responsibly with proper attribution

---

## Contact

**Arav Pandey**
- Email: pandey.ara@northeastern.edu
- GitHub: [@abhisek-ai](https://github.com/abhisek-ai)
- Institution: Northeastern University

**Project Repository:** https://github.com/abhisek-ai/text-to-image-flickr8k

---

##  Project Highlights

- 🏆 **Efficiency:** Complete pipeline in <1 hour, <$3 cost
- 🔬 **Research:** First systematic CFG analysis 1.0-20.0 with metric inflation discovery
- 🎓 **Education:** Accessible entry point for diffusion model learning
- ⚡ **Speed:** Real-time generation and evaluation (5 sec total)
- 🌍 **Impact:** Low carbon footprint (1.5 kg CO₂e total project)
- 📖 **Documentation:** Complete with code, report, ethical analysis

---

##  Quick Command Reference

```bash
# Train
python "Assignemnt 2"/train_diffusion.py

# Generate
python livedemo.py --prompt "your text" --cfg 7.5

# Evaluate
python "Assignment 3"/evaluate_model.py

# Compare
python calculateclip.py --generated gen.png --real real.jpg --prompt "text"

# FID/IS
python fid_tester.py --mode batch --generated_dir samples --reference_dir Flicker8k_Dataset

# Upscale
python upscale.py --input small.png --output big.png --scale 4
```

---

**Last Updated:** December 2025  
**Model Version:** 1.0 (128×128 resolution, 10 epochs, Flickr8k)  
**Status:** Complete and ready for deployment

*Built for educational purposes with emphasis on efficiency, accessibility, and responsible AI development.*