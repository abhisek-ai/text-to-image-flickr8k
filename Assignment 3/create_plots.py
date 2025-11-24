"""
Create plots for evaluation results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_cfg_comparison(results_file='evaluation_results/evaluation_results.json', 
                       output_dir='evaluation_results'):
    """Plot CLIP scores vs CFG values"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    cfg_values = []
    clip_scores = []
    clip_stds = []
    
    for key in sorted(results.keys()):
        if key.startswith('cfg_'):
            cfg = float(key.split('_')[1])
            cfg_values.append(cfg)
            clip_scores.append(results[key]['clip_score'])
            clip_stds.append(results[key].get('clip_std', 0))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(cfg_values, clip_scores, yerr=clip_stds, 
                marker='o', linewidth=2, markersize=8, capsize=5)
    
    ax.set_xlabel('Guidance Scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('CLIP Score', fontsize=12, fontweight='bold')
    ax.set_title('Text-Image Alignment vs Guidance Scale', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Highlight optimal
    optimal_idx = np.argmax(clip_scores)
    ax.axvline(cfg_values[optimal_idx], color='red', linestyle='--', 
               alpha=0.5, label=f'Optimal CFG={cfg_values[optimal_idx]}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cfg_comparison_plot.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/cfg_comparison_plot.png")
    
    plt.close()


def plot_sensitivity_analysis(results_file='evaluation_results/sensitivity_results.json',
                              output_dir='evaluation_results'):
    """Plot sensitivity analysis results"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    cfg_values = [r['cfg'] for r in results]
    clip_scores = [r['clip_score_mean'] for r in results]
    clip_stds = [r['clip_score_std'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(cfg_values, clip_scores, yerr=clip_stds,
                marker='s', linewidth=2, markersize=8, capsize=5, color='green')
    
    ax.set_xlabel('Guidance Scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('CLIP Score', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Sensitivity: Guidance Scale Impact', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sensitivity_plot.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/sensitivity_plot.png")
    
    plt.close()


def plot_training_loss(log_file='logs/training_log.json', output_dir='evaluation_results'):
    """Plot training loss curve"""
    
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    epochs = [log['epoch'] for log in logs]
    losses = [log['loss'] for log in logs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(len(losses)), losses, linewidth=2, color='blue')
    
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Progression', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_loss.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/training_loss.png")
    
    plt.close()


def create_comparison_grid(samples_dir='samples', output_dir='evaluation_results'):
    """Create grid comparing all CFG values"""
    from PIL import Image
    
    cfg_values = [1.0, 3.0, 7.5, 12.0, 20.0]
    prompt = "a_dog_running_in_grass"
    
    images = []
    for cfg in cfg_values:
        img_path = Path(samples_dir) / f"cfg_{cfg}_{prompt}.png"
        if img_path.exists():
            images.append(Image.open(img_path))
    
    if not images:
        print("No images found for grid")
        return
    
    fig, axes = plt.subplots(1, len(images), figsize=(20, 4))
    
    for ax, img, cfg in zip(axes, images, cfg_values):
        ax.imshow(img)
        ax.set_title(f'CFG={cfg}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Guidance Scale Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cfg_grid_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/cfg_grid_comparison.png")
    
    plt.close()


def main():
    
    print("CREATING PLOTS")
    
    
    output_dir = 'evaluation_results'
    Path(output_dir).mkdir(exist_ok=True)
    
    print("\n1. CFG comparison plot...")
    plot_cfg_comparison()
    
    print("\n2. Sensitivity analysis plot...")
    plot_sensitivity_analysis()
    
    print("\n3. Training loss plot...")
    plot_training_loss()
    
    print("\n4. Comparison grid...")
    create_comparison_grid()
    
    print("\n")
    print("ALL PLOTS CREATED!")


if __name__ == "__main__":
    main()