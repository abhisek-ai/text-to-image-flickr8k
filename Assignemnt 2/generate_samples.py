"""
Generate samples and run CFG experiments
Run this AFTER training completes
"""

import torch
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

# Import from train_diffusion.py
sys.path.append('.')
from train_diffusion import ConditionalUNet, DiffusionTrainer

def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {checkpoint_path}...")
    
    # Load CLIP
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()
    
    # Load model
    unet = ConditionalUNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    unet.load_state_dict(checkpoint['model_state_dict'])
    
    trainer = DiffusionTrainer(unet, text_encoder, device=device)
    print(f"✓ Loaded model from epoch {checkpoint['epoch']+1}, loss: {checkpoint['loss']:.4f}")
    
    return trainer, device


def generate_with_cfg(trainer, tokenizer, prompt, guidance_scales, output_dir='samples'):
    """Generate images with different CFG values"""
    os.makedirs(output_dir, exist_ok=True)
    device = trainer.device
    
    print(f"\nGenerating for prompt: '{prompt}'")
    
    # Tokenize
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    results = []
    
    for cfg in guidance_scales:
        print(f"  CFG={cfg}...", end=' ')
        
        samples = trainer.sample(
            text_inputs.input_ids,
            num_samples=1,
            guidance_scale=cfg
        )
        
        # Convert to image
        img = samples[0].cpu()
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]
        img = img.clamp(0, 1)
        img_pil = transforms.ToPILImage()(img)
        
        # Save
        safe_prompt = prompt.replace(' ', '_')[:30]
        filename = f"cfg_{cfg}_{safe_prompt}.png"
        img_pil.save(f'{output_dir}/{filename}')
        results.append((cfg, img_pil))
        print(f"✓")
    
    return results


def create_comparison_grid(results, prompt, save_path='cfg_comparison.png'):
    """Create visual comparison grid"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]
    
    for ax, (cfg, img) in zip(axes, results):
        ax.imshow(img)
        ax.set_title(f'CFG={cfg}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle(f'Prompt: {prompt[:50]}...', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison: {save_path}")


def test_noise_schedules(trainer, tokenizer, prompt, output_dir='samples'):
    """Compare different noise schedules"""
    print("\n" + "="*60)
    print("TESTING NOISE SCHEDULES")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    device = trainer.device
    
    text_inputs = tokenizer(
        prompt, 
        padding="max_length", 
        max_length=77, 
        truncation=True, 
        return_tensors="pt"
    ).to(device)
    
    results = []
    
    # Test 1: Linear (default)
    print("Testing Linear schedule...", end=' ')
    samples = trainer.sample(text_inputs.input_ids, num_samples=1, guidance_scale=7.5)
    img = (samples[0].cpu() + 1) / 2
    img = img.clamp(0, 1)
    img_pil = transforms.ToPILImage()(img)
    img_pil.save(f'{output_dir}/schedule_linear.png')
    results.append(('Linear', img_pil))
    print("✓")
    
    # Test 2: Cosine schedule
    print("Testing Cosine schedule...", end=' ')
    original_betas = trainer.betas.clone()
    
    # Cosine schedule
    timesteps = trainer.timesteps
    s = 0.008
    steps = torch.linspace(0, timesteps, timesteps + 1)
    alphas_cumprod = torch.cos(((steps / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999).to(device)
    
    trainer.betas = betas
    trainer.alphas = 1.0 - betas
    trainer.alphas_cumprod = torch.cumprod(trainer.alphas, dim=0)
    
    samples = trainer.sample(text_inputs.input_ids, num_samples=1, guidance_scale=7.5)
    img = (samples[0].cpu() + 1) / 2
    img = img.clamp(0, 1)
    img_pil = transforms.ToPILImage()(img)
    img_pil.save(f'{output_dir}/schedule_cosine.png')
    results.append(('Cosine', img_pil))
    print("✓")
    
    # Restore original
    trainer.betas = original_betas
    trainer.alphas = 1.0 - trainer.betas
    trainer.alphas_cumprod = torch.cumprod(trainer.alphas, dim=0)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (name, img) in zip(axes, results):
        ax.imshow(img)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle(f'Noise Schedule Comparison\nPrompt: {prompt[:50]}...', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/schedule_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/schedule_comparison.png")
    
    return results


def main():
    print("="*60)
    print("GENERATING SAMPLES - ASSIGNMENT 2")
    print("="*60)
    
    # Configuration
    CHECKPOINT = 'checkpoints/checkpoint_epoch_5.pt'
    OUTPUT_DIR = 'samples'
    
    # Load model
    trainer, device = load_model(CHECKPOINT)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    # Test prompts (simple ones for 64x64 resolution)
    prompts = [
        "a dog running in grass",
        "a child playing with a ball",
        "people walking on a beach"
    ]
    
    # Experiment 1: CFG Scaling
    print("\n" + "="*60)
    print("EXPERIMENT 1: CLASSIFIER-FREE GUIDANCE")
    print("="*60)
    
    guidance_scales = [1.0, 3.0, 7.5, 12.0, 20.0]
    
    # Generate for first 2 prompts
    for i, prompt in enumerate(prompts[:2]):
        results = generate_with_cfg(trainer, tokenizer, prompt, guidance_scales, OUTPUT_DIR)
        create_comparison_grid(results, prompt, f'{OUTPUT_DIR}/cfg_comparison_{i+1}.png')
    
    # Experiment 2: Noise Schedules
    print("\n" + "="*60)
    print("EXPERIMENT 2: NOISE SCHEDULES")
    print("="*60)
    
    schedule_results = test_noise_schedules(trainer, tokenizer, prompts[0], OUTPUT_DIR)
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"Results saved in: {OUTPUT_DIR}/")
    print(f"\nGenerated files:")
    print(f"  - 10 CFG comparison images (5 scales x 2 prompts)")
    print(f"  - 2 CFG comparison grids")
    print(f"  - 2 noise schedule images")
    print(f"  - 1 schedule comparison grid")
    print(f"\nTotal: ~15 images")
    print("\nNext steps:")
    print("  1. Check images in samples/")
    print("  2. Write observations.md")
    print("  3. Package for submission")


if __name__ == "__main__":
    main()