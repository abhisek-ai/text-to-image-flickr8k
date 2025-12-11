"""
Compare Generated Image vs Real Image
Calculates: CLIP Score, FID approximation, IS, Quality Metrics
Perfect for live demo!
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms, models
from pathlib import Path
import argparse
from scipy import linalg

class ImageComparator:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print("Initializing evaluation models...")
        
        # CLIP for alignment
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # InceptionV3 for FID/IS
        inception = models.inception_v3(weights='DEFAULT', transform_input=False)
        inception.eval()
        inception.fc = torch.nn.Identity()
        inception.aux_logits = False
        self.inception = inception.to(self.device)
        
        self.inception_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("✓ Models loaded\n")
    
    def calculate_clip_score(self, image, prompt):
        """CLIP score for text-image alignment"""
        inputs = self.clip_processor(
            text=[prompt],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            score = outputs.logits_per_image.item()
        
        return score
    
    def extract_inception_features(self, image):
        """Extract InceptionV3 features"""
        if isinstance(image, (str,Path)):
            image = Image.open(image).convert('RGB')
        
        img_tensor = self.inception_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.inception(img_tensor)
            features = features.squeeze().cpu().numpy()
        
        return features
    
    def calculate_feature_distance(self, features1, features2):
        """L2 distance between two feature vectors"""
        return np.linalg.norm(features1 - features2)
    
    def calculate_quality_metrics(self, image):
        """Image quality metrics"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        img_np = np.array(image) / 255.0
        
        # Convert to grayscale for sharpness
        gray = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
        
        # Sharpness (Laplacian variance)
        from scipy.ndimage import laplace
        laplacian = laplace(gray)
        sharpness = np.var(laplacian)
        
        # Contrast
        contrast = np.std(img_np)
        
        # Color diversity
        color_diversity = np.std([img_np[:,:,i].mean() for i in range(3)])
        
        # Brightness
        brightness = np.mean(img_np)
        
        return {
            'sharpness': float(sharpness),
            'contrast': float(contrast),
            'color_diversity': float(color_diversity),
            'brightness': float(brightness)
        }
    
    def compare_images(self, generated_path, real_path, prompt):
        """
        Complete comparison of generated vs real image
        """
        print("="*60)
        print("IMAGE COMPARISON")
        print("="*60)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated_path}")
        print(f"Real: {real_path}\n")
        
        # Load images
        gen_img = Image.open(generated_path).convert('RGB')
        real_img = Image.open(real_path).convert('RGB')
        
        print(f"Generated size: {gen_img.size}")
        print(f"Real size: {real_img.size}\n")
        
        results = {}
        
        # 1. CLIP Scores
        print("[1/4] Calculating CLIP Scores...")
        gen_clip = self.calculate_clip_score(gen_img, prompt)
        real_clip = self.calculate_clip_score(real_img, prompt)
        
        print(f"  Generated CLIP: {gen_clip:.3f}")
        print(f"  Real CLIP: {real_clip:.3f}")
        print(f"  Difference: {abs(gen_clip - real_clip):.3f}")
        
        results['clip_generated'] = gen_clip
        results['clip_real'] = real_clip
        results['clip_diff'] = abs(gen_clip - real_clip)
        
        # 2. InceptionV3 Feature Distance
        print("\n[2/4] Calculating Feature Distance (FID-like)...")
        gen_features = self.extract_inception_features(gen_img)
        real_features = self.extract_inception_features(real_img)
        
        feature_dist = self.calculate_feature_distance(gen_features, real_features)
        print(f"  Feature Distance: {feature_dist:.2f}")
        print(f"  (Lower = more similar. Typical: 10-50 for similar, >100 for very different)")
        
        results['feature_distance'] = feature_dist
        
        # 3. Quality Metrics Comparison
        print("\n[3/4] Comparing Quality Metrics...")
        gen_quality = self.calculate_quality_metrics(gen_img)
        real_quality = self.calculate_quality_metrics(real_img)
        
        print("\n  Metric          | Generated | Real     | Difference")
        print("  " + "-"*56)
        for key in gen_quality.keys():
            print(f"  {key:15} | {gen_quality[key]:9.3f} | {real_quality[key]:8.3f} | {abs(gen_quality[key]-real_quality[key]):10.3f}")
        
        results['quality_generated'] = gen_quality
        results['quality_real'] = real_quality
        
        # 4. Overall Assessment
        print("\n[4/4] Overall Assessment")
        print("  " + "-"*56)
        
        # CLIP alignment
        if gen_clip > 25:
            clip_rating = "Excellent"
        elif gen_clip > 22:
            clip_rating = "Good"
        elif gen_clip > 20:
            clip_rating = "Fair"
        else:
            clip_rating = "Poor"
        
        print(f"  Generated Alignment: {clip_rating}")
        
        # Feature similarity
        if feature_dist < 30:
            similarity = "Very Similar"
        elif feature_dist < 60:
            similarity = "Moderately Similar"
        elif feature_dist < 100:
            similarity = "Somewhat Different"
        else:
            similarity = "Very Different"
        
        print(f"  Feature Similarity: {similarity}")
        
        # Winner
        if gen_clip > real_clip:
            print(f"\n  🏆 Generated image has BETTER text alignment than real!")
            print(f"     (Generated: {gen_clip:.2f} vs Real: {real_clip:.2f})")
        else:
            print(f"\n  Real image has better alignment (as expected)")
            print(f"     (Real: {real_clip:.2f} vs Generated: {gen_clip:.2f})")
        
        print("\n" + "="*60 + "\n")
        
        return results


def compare_to_reference_distribution(generated_path, reference_dir, prompt):
    """
    Compare single generated image to entire reference distribution
    More comprehensive than 1-to-1 comparison
    """
    print("="*60)
    print("COMPARING TO REFERENCE DISTRIBUTION")
    print("="*60)
    print(f"\nGenerated: {generated_path}")
    print(f"Reference: {reference_dir}")
    print(f"Prompt: '{prompt}'\n")
    
    comparator = ImageComparator()
    
    # Load generated image
    gen_img = Image.open(generated_path).convert('RGB')
    gen_clip = comparator.calculate_clip_score(gen_img, prompt)
    gen_features = comparator.extract_inception_features(gen_img)
    
    print(f"Generated CLIP Score: {gen_clip:.3f}")
    
    # Load reference images
    print("\nLoading reference images...")
    ref_paths = list(Path(reference_dir).glob('*.jpg'))[:100]  # Sample 100 for speed
    print(f"Loaded {len(ref_paths)} reference images")
    
    # Calculate reference CLIP scores
    print("\nCalculating reference CLIP scores...")
    ref_clips = []
    for ref_path in ref_paths[:20]:  # Sample 20 for speed
        ref_img = Image.open(ref_path).convert('RGB')
        ref_clip = comparator.calculate_clip_score(ref_img, prompt)
        ref_clips.append(ref_clip)
    
    ref_clip_mean = np.mean(ref_clips)
    ref_clip_std = np.std(ref_clips)
    
    print(f"Reference CLIP (mean): {ref_clip_mean:.3f} ± {ref_clip_std:.3f}")
    
    # Extract reference features
    print("\nExtracting reference features...")
    ref_features_list = []
    for ref_path in ref_paths:
        ref_feat = comparator.extract_inception_features(ref_path)
        ref_features_list.append(ref_feat)
    
    ref_features = np.array(ref_features_list)
    ref_mean = np.mean(ref_features, axis=0)
    
    # Distance from reference mean
    dist_from_mean = np.linalg.norm(gen_features - ref_mean)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nCLIP Score:")
    print(f"  Generated: {gen_clip:.3f}")
    print(f"  Reference Mean: {ref_clip_mean:.3f} ± {ref_clip_std:.3f}")
    if gen_clip > ref_clip_mean:
        print(f"  ✓ Generated ABOVE average!")
    else:
        print(f"  Generated below average by {ref_clip_mean - gen_clip:.2f}")
    
    print(f"\nFeature Distance from Reference:")
    print(f"  Distance: {dist_from_mean:.2f}")
    print(f"  (Lower = closer to real images)")
    
    print(f"\n{'='*60}\n")
    
    return {
        'gen_clip': gen_clip,
        'ref_clip_mean': ref_clip_mean,
        'ref_clip_std': ref_clip_std,
        'feature_distance': dist_from_mean
    }


def main():
    parser = argparse.ArgumentParser(description='Compare Generated vs Real Images')
    parser.add_argument('--generated', type=str, required=True, help='Generated image path')
    parser.add_argument('--real', type=str, help='Real image path (for 1-to-1 comparison)')
    parser.add_argument('--reference_dir', type=str, help='Reference directory (for distribution comparison)')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    comparator = ImageComparator(device=args.device)
    
    if args.real:
        # 1-to-1 comparison
        results = comparator.compare_images(args.generated, args.real, args.prompt)
    elif args.reference_dir:
        # Distribution comparison
        results = compare_to_reference_distribution(args.generated, args.reference_dir, args.prompt)
    else:
        print("ERROR: Must provide either --real or --reference_dir")
        return
    
    print("Comparison complete!")


if __name__ == "__main__":
    print("""
Image Comparison Script
=======================

Usage:

1. Compare generated vs specific real image:
   python compare_image.py \\
       --generated demo_vs_real.png \\
       --real Flicker8k_Dataset/1000268201_693b08cb0e.jpg \\
       --prompt "a child in a pink dress climbing stairs"

2. Compare generated vs entire reference distribution:
   python compare_image.py \\
       --generated demo_vs_real.png \\
       --reference_dir Flicker8k_Dataset \\
       --prompt "a child in a pink dress climbing stairs"

Metrics Calculated:
- CLIP Score (text-image alignment)
- InceptionV3 Feature Distance (visual similarity)
- Quality Metrics (sharpness, contrast, color)
- FID-like approximation
    """)
    
    import sys
    if len(sys.argv) > 1:
        main()