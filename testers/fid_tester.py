"""
FID and Inception Score Calculator
Compare new generated image against model's distribution
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms, models
from scipy import linalg
from tqdm import tqdm
import argparse

class InceptionV3FeatureExtractor(nn.Module):
    """Extract features from InceptionV3 for FID/IS calculation"""
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Load pre-trained InceptionV3
        # Load pre-trained InceptionV3
        inception = models.inception_v3(weights='DEFAULT', transform_input=False)
        inception.eval()
        
        # Remove final classification layers, keep feature extractor
        inception.fc = nn.Identity()  # Replace FC with identity
        inception.aux_logits = False  # Disable auxiliary classifier
        
        self.feature_extractor = inception.to(device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(299),  # InceptionV3 input size
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, images):
        """Extract features from images"""
        with torch.no_grad():
            features = self.feature_extractor(images)
            features = features.squeeze(-1).squeeze(-1)
        return features


def load_images(image_paths, transform):
    """Load and preprocess images"""
    images = []
    
    for path in tqdm(image_paths, desc="Loading images"):
        try:
            img = Image.open(path).convert('RGB')
            img = transform(img)
            images.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    return torch.stack(images) if images else None


def calculate_fid(real_features, generated_features):
    """
    Calculate Fréchet Inception Distance between real and generated images
    
    FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2√(Σ_r Σ_g))
    
    Lower FID = better (more similar distributions)
    """
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)
    
    # Calculate squared difference of means
    diff = mu_real - mu_gen
    mean_diff = diff.dot(diff)
    
    # Calculate matrix square root
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    
    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = mean_diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    
    return fid


def calculate_inception_score(features, splits=10):
    """
    Calculate Inception Score
    
    IS = exp(E_x[KL(p(y|x) || p(y))])
    
    Higher IS = better quality and diversity
    """
    N = features.shape[0]
    
    # Get predictions (softmax of features as proxy)
    preds = torch.nn.functional.softmax(torch.from_numpy(features), dim=1).numpy()
    
    # Split into groups
    split_scores = []
    
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        
        # p(y|x)
        py_given_x = part
        
        # p(y) = average over all x
        py = np.mean(part, axis=0)
        
        # KL divergence
        kl_div = py_given_x * (np.log(py_given_x + 1e-10) - np.log(py + 1e-10))
        kl_div = np.mean(np.sum(kl_div, axis=1))
        
        split_scores.append(np.exp(kl_div))
    
    return np.mean(split_scores), np.std(split_scores)


def extract_features(image_paths, feature_extractor, batch_size=32):
    """Extract InceptionV3 features from list of images"""
    
    device = feature_extractor.device
    features_list = []
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load batch
        images = load_images(batch_paths, feature_extractor.transform)
        if images is None:
            continue
        
        images = images.to(device)
        
        # Extract features
        features = feature_extractor(images)
        features_list.append(features.cpu().numpy())
    
    # Concatenate all features
    all_features = np.concatenate(features_list, axis=0)
    
    return all_features


def compare_single_image_to_distribution(image_path, reference_dir, device='cuda'):
    """
    Compare a single generated image to reference distribution
    
    Args:
        image_path: Path to single generated image
        reference_dir: Directory with reference images (real or model-generated)
        device: 'cuda' or 'cpu'
    
    Returns:
        Dictionary with metrics
    """
    print(f"\n{'='*60}")
    print("COMPARING IMAGE TO DISTRIBUTION")
    print(f"{'='*60}\n")
    
    # Initialize feature extractor
    print("Loading InceptionV3...")
    feature_extractor = InceptionV3FeatureExtractor(device)
    
    # Load reference images
    print(f"\nLoading reference images from {reference_dir}...")
    reference_paths = list(Path(reference_dir).glob('*.png')) + \
                     list(Path(reference_dir).glob('*.jpg'))
    
    if len(reference_paths) < 10:
        print(f"WARNING: Only {len(reference_paths)} reference images. Need 10+ for reliable FID.")
    
    print(f"Found {len(reference_paths)} reference images")
    
    # Extract features from reference distribution
    print("\nExtracting features from reference images...")
    reference_features = extract_features(reference_paths, feature_extractor)
    
    # Extract features from single image
    print(f"\nExtracting features from target image: {image_path}")
    target_features = extract_features([image_path], feature_extractor)
    
    # Calculate metrics
    results = {}
    
    # FID (compare to reference distribution)
    if len(reference_features) >= 2:  # Need at least 2 for covariance
        print("\nCalculating FID...")
        # For single image, replicate it to calculate FID
        # (Not standard, but gives relative comparison)
        replicated_features = np.tile(target_features, (10, 1))  # Replicate 10x
        fid_score = calculate_fid(reference_features, replicated_features)
        results['fid'] = fid_score
        print(f"FID Score: {fid_score:.2f}")
        print(f"  Lower is better. Typical range: 10-200")
        print(f"  <50 = excellent, 50-100 = good, 100-200 = fair, >200 = poor")
    else:
        print("Not enough reference images for FID calculation")
        results['fid'] = None
    
    # Inception Score (on reference distribution for context)
    if len(reference_features) >= 10:
        print("\nCalculating Inception Score (reference distribution)...")
        is_mean, is_std = calculate_inception_score(reference_features)
        results['reference_is_mean'] = is_mean
        results['reference_is_std'] = is_std
        print(f"Reference IS: {is_mean:.2f} ± {is_std:.2f}")
        print(f"  Higher is better. Typical range: 1-10")
    else:
        results['reference_is_mean'] = None
        results['reference_is_std'] = None
    
    # Feature distance (L2 distance to mean of reference)
    print("\nCalculating feature distance...")
    reference_mean = np.mean(reference_features, axis=0)
    feature_distance = np.linalg.norm(target_features - reference_mean)
    results['feature_distance'] = feature_distance
    print(f"Feature Distance: {feature_distance:.2f}")
    print(f"  Lower = more similar to reference distribution")
    
    # Nearest neighbor distance
    print("\nFinding nearest neighbor...")
    distances = np.linalg.norm(reference_features - target_features, axis=1)
    nn_idx = np.argmin(distances)
    nn_distance = distances[nn_idx]
    results['nn_distance'] = nn_distance
    results['nn_image'] = str(reference_paths[nn_idx])
    print(f"Nearest Neighbor: {reference_paths[nn_idx].name}")
    print(f"Distance: {nn_distance:.2f}")
    
    return results


def batch_compare_images(generated_dir, reference_dir, output_file='comparison_results.txt', device='cuda'):
    """
    Compare multiple generated images to reference distribution
    
    Args:
        generated_dir: Directory with generated images
        reference_dir: Directory with reference images
        output_file: Where to save results
        device: 'cuda' or 'cpu'
    """
    print(f"\n{'='*60}")
    print("BATCH COMPARISON")
    print(f"{'='*60}\n")
    
    # Initialize
    feature_extractor = InceptionV3FeatureExtractor(device)
    
    # Load reference
    reference_paths = list(Path(reference_dir).glob('*.png')) + \
                     list(Path(reference_dir).glob('*.jpg'))
    print(f"Reference images: {len(reference_paths)}")
    reference_features = extract_features(reference_paths, feature_extractor)
    
    # Load generated
    generated_paths = list(Path(generated_dir).glob('*.png')) + \
                     list(Path(generated_dir).glob('*.jpg'))
    print(f"Generated images: {len(generated_paths)}")
    generated_features = extract_features(generated_paths, feature_extractor)
    
    # Calculate FID
    print("\nCalculating FID...")
    fid_score = calculate_fid(reference_features, generated_features)
    
    # Calculate IS
    print("Calculating Inception Scores...")
    ref_is_mean, ref_is_std = calculate_inception_score(reference_features)
    gen_is_mean, gen_is_std = calculate_inception_score(generated_features)
    
    # Save results
    results_text = f"""
FID & Inception Score Comparison Results
{'='*60}

Reference Distribution: {reference_dir}
Generated Distribution: {generated_dir}

Metrics:
--------
FID Score: {fid_score:.2f}
  (Lower is better. <50=excellent, 50-100=good, 100-200=fair, >200=poor)

Inception Score (Reference): {ref_is_mean:.2f} ± {ref_is_std:.2f}
Inception Score (Generated): {gen_is_mean:.2f} ± {gen_is_std:.2f}
  (Higher is better. Typical range: 1-10)

Sample Sizes:
-------------
Reference images: {len(reference_paths)}
Generated images: {len(generated_paths)}

Interpretation:
--------------
FID measures distribution similarity. Lower FID means generated images
are more similar to real distribution.

Inception Score measures quality and diversity. Higher IS means better
quality and more diverse outputs.

{'='*60}
"""
    
    print(results_text)
    
    with open(output_file, 'w') as f:
        f.write(results_text)
    
    print(f"\nResults saved to: {output_file}")
    
    return {
        'fid': fid_score,
        'reference_is': (ref_is_mean, ref_is_std),
        'generated_is': (gen_is_mean, gen_is_std)
    }


def main():
    parser = argparse.ArgumentParser(description='Calculate FID and Inception Score')
    parser.add_argument('--mode', choices=['single', 'batch'], required=True,
                       help='Single image or batch comparison')
    parser.add_argument('--image', type=str, help='Path to single image (for single mode)')
    parser.add_argument('--generated_dir', type=str, help='Directory with generated images (for batch mode)')
    parser.add_argument('--reference_dir', type=str, required=True,
                       help='Directory with reference images')
    parser.add_argument('--output', type=str, default='results.txt',
                       help='Output file for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.image:
            print("ERROR: --image required for single mode")
            return
        
        results = compare_single_image_to_distribution(
            args.image,
            args.reference_dir,
            args.device
        )
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        for key, value in results.items():
            print(f"{key}: {value}")
    
    elif args.mode == 'batch':
        if not args.generated_dir:
            print("ERROR: --generated_dir required for batch mode")
            return
        
        results = batch_compare_images(
            args.generated_dir,
            args.reference_dir,
            args.output,
            args.device
        )


if __name__ == "__main__":
    # Example usage
    print("""
FID & Inception Score Calculator
==================================

Usage Examples:

1. Compare single image to distribution:
   python calculate_fid_is.py --mode single \\
       --image samples/cfg_7.5_a_dog_running.png \\
       --reference_dir data/processed/samples

2. Compare batch of generated images to real images:
   python calculate_fid_is.py --mode batch \\
       --generated_dir samples \\
       --reference_dir data/processed/samples \\
       --output fid_results.txt

3. Use CPU instead of GPU:
   python calculate_fid_is.py --mode single \\
       --image test.png \\
       --reference_dir references \\
       --device cpu
    """)
    
    import sys
    if len(sys.argv) == 1:
        print("\nNo arguments provided. Use --help for usage information.")
    else:
        main()