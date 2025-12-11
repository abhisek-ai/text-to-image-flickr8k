"""
Live Demo Script for Presentation
Generate image from prompt + Calculate metrics in real-time
"""

import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor
from torchvision import transforms, models
import sys
import numpy as np
from PIL import Image
from pathlib import Path
import time

# Import your model
sys.path.insert(0, 'Assignemnt 2')
from train_diffusion import ConditionalUNet, DiffusionTrainer

class LiveDemo:
    def __init__(self, checkpoint_path='checkpoints/checkpoint_epoch_10.pt'):
        print("="*60)
        print("INITIALIZING LIVE DEMO")
        print("="*60)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load model
        print("\n1. Loading diffusion model...")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        unet = ConditionalUNet().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        unet.load_state_dict(checkpoint['model_state_dict'])
        
        self.trainer = DiffusionTrainer(unet, text_encoder, device=self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        print(f"   ✓ Loaded model from epoch {checkpoint['epoch']+1}")
        
        # Load CLIP for scoring
        print("\n2. Loading CLIP for evaluation...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("   ✓ CLIP loaded")
        
        print("\n" + "="*60)
        print("READY FOR LIVE DEMO!")
        print("="*60 + "\n")
    
    def generate_and_evaluate(self, prompt, guidance_scale=7.5, save_path=None,calculateclip=None):
        """
        Generate image from prompt and calculate metrics in real-time
        """
        print(f"\n{'='*60}")
        print(f"PROMPT: '{prompt}'")
        print(f"CFG: {guidance_scale}")
        print(f"{'='*60}")
        
        # 1. Generate image
        print("\n[1/4] Generating image...")
        start_time = time.time()
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        samples = self.trainer.sample(
            text_inputs.input_ids,
            num_samples=1,
            img_size = 128,
            guidance_scale=guidance_scale
        )
        
        generation_time = time.time() - start_time
        print(f"   ✓ Generated in {generation_time:.1f} seconds")
        
        # Convert to PIL
        img_tensor = samples[0].cpu()
        img_tensor = (img_tensor + 1) / 2  # [-1, 1] -> [0, 1]
        img_tensor = img_tensor.clamp(0, 1)
        img_pil = transforms.ToPILImage()(img_tensor)
        
        # Save if requested
        if save_path:
            img_pil.save(save_path)
            print(f"   ✓ Saved to: {save_path}")
        
        # 2. Calculate CLIP Score
        print("\n[2/4] Calculating CLIP Score (text-image alignment)...")
        clip_score = self.calculate_clip_score(img_pil, prompt)
        print(f"   ✓ CLIP Score: {clip_score:.3f}")
        
        # 3. Calculate image quality metrics
        print("\n[3/4] Calculating image quality metrics...")
        quality_metrics = self.calculate_quality_metrics(img_tensor)
        print(f"   ✓ Sharpness: {quality_metrics['sharpness']:.3f}")
        print(f"   ✓ Contrast: {quality_metrics['contrast']:.3f}")
        print(f"   ✓ Color Diversity: {quality_metrics['color_diversity']:.3f}")
        
        if calculateclip:
            print("\n[BONUS] Comparing to real image...")
            feat_dist = self.calculate_feature_distance(img_pil, calculateclip)
            print(f"   ✓ Feature Distance: {feat_dist:.2f}")
            if feat_dist < 30:
                print(f"   Assessment: Very Similar")
            elif feat_dist < 60:
                print(f"   Assessment: Moderately Similar")
            else:
                print(f"   Assessment: Different")
            results['feature_distance'] = feat_dist
        
        # 4. Summary
        print("\n[4/4] Summary")
        print("   " + "-"*56)
        
        # Interpret CLIP score
        if clip_score > 25:
            alignment = "Excellent"
        elif clip_score > 22:
            alignment = "Good"
        elif clip_score > 20:
            alignment = "Fair"
        else:
            alignment = "Poor"
        
        print(f"   Text-Image Alignment: {alignment}")
        print(f"   Generation Speed: {generation_time:.1f}s (1000 steps)")
        print(f"   Resolution: 64×64 pixels")
        
        results = {
            'prompt': prompt,
            'guidance_scale': guidance_scale,
            'clip_score': clip_score,
            'generation_time': generation_time,
            'quality_metrics': quality_metrics,
            'image': img_pil
        }
        
        print(f"\n{'='*60}\n")
        
        return results
    
    def calculate_clip_score(self, image, prompt):
        """Calculate CLIP score for text-image alignment"""
        inputs = self.clip_processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = logits_per_image.item()
        
        return score
    
    def calculate_quality_metrics(self, img_tensor):
        """Calculate basic image quality metrics"""
        img_np = img_tensor.numpy()
        
        # Sharpness (variance of Laplacian)
        gray = 0.299 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]
        laplacian = np.gradient(np.gradient(gray)[0])[0]
        sharpness = np.var(laplacian)
        
        # Contrast (std deviation)
        contrast = np.std(img_np)
        
        # Color diversity (std of channel means)
        color_diversity = np.std([img_np[i].mean() for i in range(3)])
        
        return {
            'sharpness': float(sharpness),
            'contrast': float(contrast),
            'color_diversity': float(color_diversity)
        }
    
    def compare_multiple_cfg(self, prompt, cfg_values=[1.0, 3.0, 7.5, 12.0, 20.0]):
        """
        Generate and compare multiple CFG values for same prompt
        Useful for showing CFG behavior during presentation
        """
        print(f"\n{'='*60}")
        print(f"CFG COMPARISON FOR: '{prompt}'")
        print(f"{'='*60}\n")
        
        results = []
        
        for cfg in cfg_values:
            result = self.generate_and_evaluate(prompt, guidance_scale=cfg, 
                                                save_path=f"demo_cfg_{cfg}.png")
            results.append(result)
            
            # Brief summary
            print(f"CFG={cfg}: CLIP={result['clip_score']:.2f}, Time={result['generation_time']:.1f}s\n")
        
        # Find best
        best_idx = np.argmax([r['clip_score'] for r in results])
        print(f"\nBest CFG: {cfg_values[best_idx]} (CLIP Score: {results[best_idx]['clip_score']:.2f})")
        
        return results
    
    def calculate_feature_distance(self, image, reference_image_path):
        """
        Calculate InceptionV3 feature distance between generated and real image
        FID-like metric for single image comparison
        """
        from torchvision.models import inception_v3
        
        # Load InceptionV3
        inception = inception_v3(weights='DEFAULT', transform_input=False)
        inception.eval()
        inception.fc = torch.nn.Identity()
        inception.aux_logits = False
        inception = inception.to(self.device)
        
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Generated features
        gen_tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            gen_feat = inception(gen_tensor).squeeze().cpu().numpy()
        
        # Real features
        real_img = Image.open(reference_image_path).convert('RGB')
        real_tensor = transform(real_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            real_feat = inception(real_tensor).squeeze().cpu().numpy()
        
        # Distance
        distance = np.linalg.norm(gen_feat - real_feat)
        
        return distance


def main():
    """
    Main demo script - modify prompts for your presentation
    """
    
    # Initialize
    demo = LiveDemo(checkpoint_path='Assignemnt 2/checkpoints/checkpoint_epoch_10.pt')
    
    # Example 1: Single prompt evaluation
    print("\n" + "="*60)
    print("DEMO 1: SINGLE PROMPT EVALUATION")
    print("="*60)
    
    result = demo.generate_and_evaluate(
        prompt="a dog running in grass",
        guidance_scale=7.5,
        save_path="demo_output.png"
    )
    
    # Example 2: CFG comparison
    print("\n" + "="*60)
    print("DEMO 2: CFG SCALE COMPARISON")
    print("="*60)
    
    cfg_results = demo.compare_multiple_cfg(
        prompt="a child playing with a ball",
        cfg_values=[3.0, 7.5, 12.0]
    )
    
    # Example 3: Professor's custom prompt (modify during presentation)
    print("\n" + "="*60)
    print("DEMO 3: CUSTOM PROMPT")
    print("="*60)
    print("Enter custom prompt (or press Enter for default): ")
    custom_prompt = input().strip()
    
    if not custom_prompt:
        custom_prompt = "people walking on a beach"
    
    custom_result = demo.generate_and_evaluate(
        prompt=custom_prompt,
        guidance_scale=7.5,
        save_path="demo_custom.png"
    )
    
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print(f"\nGenerated images:")
    print(f"  - demo_output.png")
    print(f"  - demo_cfg_*.png (CFG comparison)")
    print(f"  - demo_custom.png")
    print("\nAll metrics calculated in real-time!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Live Demo for Presentation')
    parser.add_argument('--prompt', type=str, help='Custom prompt to generate')
    parser.add_argument('--cfg', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--output', type=str, default='demo_output.png', help='Output path')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive or args.prompt:
        demo = LiveDemo()
        
        if args.prompt:
            # Single generation from command line
            result = demo.generate_and_evaluate(
                prompt=args.prompt,
                guidance_scale=args.cfg,
                save_path=args.output
            )
        else:
            # Interactive mode
            main()
    else:
        # Show usage
        print("""
Live Demo Script for Presentation
==================================

Usage:

1. Quick generation:
   python live_demo.py --prompt "a dog running" --cfg 7.5

2. Interactive mode:
   python live_demo.py --interactive

3. Full demo sequence:
   python live_demo.py
        """)