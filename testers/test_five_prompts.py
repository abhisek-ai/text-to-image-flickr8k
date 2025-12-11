#!/usr/bin/env python3
"""
Test 5 Prompts for Milestone 1
This script generates embeddings for 5 test prompts and creates visualizations
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor
from PIL import Image
import json
from datetime import datetime
from pathlib import Path

class TestPromptsPipeline:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load CLIP model
        self.model_name = "openai/clip-vit-base-patch32"
        print(f"Loading CLIP model: {self.model_name}")
        
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        print("✅ Model loaded successfully!")
        
    def encode_text(self, text):
        """Convert text to CLIP embedding"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def load_flickr_captions(self):
        """Load preprocessed Flickr8k captions"""
        # Try different possible paths
        paths_to_try = [
            "data/processed/captions_clean.csv",
            "data/processed/captions.csv",
            "data/captions.csv",
            "captions.csv"
        ]
        
        for path in paths_to_try:
            if Path(path).exists():
                print(f"Loading captions from {path}")
                df = pd.read_csv(path)
                # Get unique captions (first caption per image if multiple)
                if 'caption_clean' in df.columns:
                    return df['caption_clean'].unique()[:1000]
                elif 'caption' in df.columns:
                    return df['caption'].unique()[:1000]
        
        print("Warning: No captions file found, using dummy captions")
        return None
    
    def test_five_prompts(self):
        """Test 5 required prompts for milestone 1"""
        
        # The 5 test prompts
        test_prompts = [
            "a dog playing in the park",
            "sunset over the ocean with waves",
            "a child eating ice cream happily",
            "people walking in a busy street",
            "mountains covered with snow"
        ]
        
        print("\n" + "="*60)
        print("TESTING 5 SAMPLE PROMPTS FOR MILESTONE 1")
        print("="*60)
        
        # Store results
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "device": self.device,
            "prompts": []
        }
        
        # Process each prompt
        all_embeddings = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[Test {i}/5] Processing: '{prompt}'")
            
            # Generate embedding
            embedding = self.encode_text(prompt)
            all_embeddings.append(embedding)
            
            # Calculate statistics
            embedding_stats = {
                "prompt": prompt,
                "embedding_shape": embedding.shape,
                "embedding_dim": embedding.shape[-1],
                "min_value": float(embedding.min()),
                "max_value": float(embedding.max()),
                "mean_value": float(embedding.mean()),
                "std_value": float(embedding.std()),
                "norm": float(np.linalg.norm(embedding))
            }
            
            results["prompts"].append(embedding_stats)
            
            # Print statistics
            print(f"  ✅ Embedding shape: {embedding.shape}")
            print(f"  📊 Statistics:")
            print(f"     - Dimension: {embedding.shape[-1]}")
            print(f"     - Norm: {embedding_stats['norm']:.4f}")
            print(f"     - Mean: {embedding_stats['mean_value']:.4f}")
            print(f"     - Std: {embedding_stats['std_value']:.4f}")
        
        # Save embeddings
        embeddings_array = np.vstack(all_embeddings)
        np.save('results/test_embeddings.npy', embeddings_array)
        print(f"\n💾 Saved embeddings to results/test_embeddings.npy")
        
        # Save results as JSON
        with open('results/test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Saved results to results/test_results.json")
        
        return embeddings_array, results
    
    def visualize_embeddings(self, embeddings, prompts):
        """Create visualization of the 5 test embeddings"""
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Embedding magnitudes
        plt.subplot(2, 3, 1)
        magnitudes = np.linalg.norm(embeddings, axis=1)
        colors = plt.cm.viridis(np.linspace(0, 1, len(prompts)))
        bars = plt.bar(range(len(prompts)), magnitudes, color=colors)
        plt.xlabel('Prompt Index')
        plt.ylabel('Embedding Magnitude')
        plt.title('Embedding Magnitudes for 5 Test Prompts')
        plt.xticks(range(len(prompts)), [f"P{i+1}" for i in range(len(prompts))])
        
        # Add value labels on bars
        for bar, mag in zip(bars, magnitudes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mag:.2f}', ha='center', va='bottom')
        
        # 2. Similarity matrix
        plt.subplot(2, 3, 2)
        # Compute cosine similarity
        similarity_matrix = np.dot(embeddings, embeddings.T)
        im = plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.title('Cosine Similarity Matrix')
        plt.xlabel('Prompt Index')
        plt.ylabel('Prompt Index')
        plt.xticks(range(len(prompts)), [f"P{i+1}" for i in range(len(prompts))])
        plt.yticks(range(len(prompts)), [f"P{i+1}" for i in range(len(prompts))])
        
        # Add values to heatmap
        for i in range(len(prompts)):
            for j in range(len(prompts)):
                plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=8)
        
        # 3. First 2 principal components
        plt.subplot(2, 3, 3)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   c=colors, s=200, alpha=0.6, edgecolors='black', linewidth=2)
        
        # Add labels
        for i, txt in enumerate([f"P{i+1}" for i in range(len(prompts))]):
            plt.annotate(txt, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=12, ha='center', va='center')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        plt.title('PCA Projection of Embeddings')
        plt.grid(True, alpha=0.3)
        
        # 4. Embedding statistics
        plt.subplot(2, 3, 4)
        stats_data = []
        for emb in embeddings:
            stats_data.append([emb.mean(), emb.std(), emb.min(), emb.max()])
        stats_data = np.array(stats_data)
        
        x = np.arange(len(prompts))
        width = 0.2
        
        plt.bar(x - 1.5*width, stats_data[:, 0], width, label='Mean', color='blue', alpha=0.7)
        plt.bar(x - 0.5*width, stats_data[:, 1], width, label='Std', color='green', alpha=0.7)
        plt.bar(x + 0.5*width, stats_data[:, 2], width, label='Min', color='red', alpha=0.7)
        plt.bar(x + 1.5*width, stats_data[:, 3], width, label='Max', color='orange', alpha=0.7)
        
        plt.xlabel('Prompt Index')
        plt.ylabel('Value')
        plt.title('Embedding Statistics')
        plt.xticks(x, [f"P{i+1}" for i in range(len(prompts))])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Prompt text display
        plt.subplot(2, 3, 5)
        plt.axis('off')
        prompt_text = "TEST PROMPTS:\n\n"
        for i, prompt in enumerate(prompts, 1):
            prompt_text += f"P{i}: {prompt}\n\n"
        plt.text(0.1, 0.9, prompt_text, fontsize=10, verticalalignment='top',
                transform=plt.gca().transAxes, wrap=True)
        plt.title('Prompt Descriptions', fontsize=12, pad=20)
        
        # 6. Summary metrics
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Calculate average similarity
        upper_triangle = np.triu_indices_from(similarity_matrix, k=1)
        avg_similarity = similarity_matrix[upper_triangle].mean()
        
        summary_text = f"""SUMMARY METRICS:
        
Model: CLIP ViT-B/32
Embedding Dimension: {embeddings.shape[1]}
Number of Test Prompts: {len(prompts)}

Average Pairwise Similarity: {avg_similarity:.4f}
Max Similarity: {similarity_matrix[upper_triangle].max():.4f}
Min Similarity: {similarity_matrix[upper_triangle].min():.4f}

Status: ✅ All prompts successfully encoded
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        
        plt.text(0.1, 0.9, summary_text, fontsize=11, verticalalignment='top',
                transform=plt.gca().transAxes, family='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.suptitle('Milestone 1: Text-to-Embedding Pipeline Results', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('results/milestone1_test_prompts.png', dpi=150, bbox_inches='tight')
        print(f"📊 Saved visualization to results/milestone1_test_prompts.png")
        
        plt.show()
        
        return fig
    
    def generate_report(self, results):
        """Generate a markdown report for the submission"""
        
        report = f"""# Milestone 1: Test Prompts Results

## Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Model Configuration
- **Model**: {results['model']}
- **Device**: {results['device']}
- **Framework**: PyTorch with Hugging Face Transformers

## 5 Test Prompts Results

"""
        
        for i, prompt_result in enumerate(results['prompts'], 1):
            report += f"""### Test {i}: "{prompt_result['prompt']}"
- ✅ Successfully generated embedding
- Embedding dimension: {prompt_result['embedding_dim']}
- Norm: {prompt_result['norm']:.4f}
- Mean: {prompt_result['mean_value']:.4f}
- Std: {prompt_result['std_value']:.4f}

"""
        
        report += """## Files Generated
1. `results/test_embeddings.npy` - NumPy array of embeddings
2. `results/test_results.json` - Detailed results in JSON format
3. `results/milestone1_test_prompts.png` - Visualization of results

## Conclusion
All 5 test prompts have been successfully processed through the text-to-embedding pipeline using CLIP.
The embeddings are ready for the next phase of the project (image generation).

## Next Steps
- Implement diffusion model for image generation
- Train on full Flickr8k dataset
- Implement evaluation metrics
"""
        
        # Save report
        with open('results/milestone1_report.md', 'w') as f:
            f.write(report)
        print(f"📝 Saved report to results/milestone1_report.md")
        
        return report


def main():
    """Main execution function"""
    print("="*60)
    print("MILESTONE 1: TESTING 5 SAMPLE PROMPTS")
    print("="*60)
    
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Initialize pipeline
    pipeline = TestPromptsPipeline()
    
    # Define the 5 test prompts
    test_prompts = [
        "a dog playing in the park",
        "sunset over the ocean with waves",
        "a child eating ice cream happily",
        "people walking in a busy street",
        "mountains covered with snow"
    ]
    
    # Run the test
    embeddings, results = pipeline.test_five_prompts()
    
    # Create visualizations
    pipeline.visualize_embeddings(embeddings, test_prompts)
    
    # Generate report
    report = pipeline.generate_report(results)
    
    print("\n" + "="*60)
    print("✅ MILESTONE 1 COMPLETE!")
    print("="*60)
    print("\nAll 5 test prompts have been successfully processed!")
    print("Check the 'results/' folder for:")
    print("  - test_embeddings.npy (embeddings array)")
    print("  - test_results.json (detailed results)")
    print("  - milestone1_test_prompts.png (visualization)")
    print("  - milestone1_report.md (submission report)")
    
    return embeddings, results


if __name__ == "__main__":
    # Run the test
    embeddings, results = main()
    
    # Print summary
    print("\n" + "="*60)
    print("READY FOR SUBMISSION!")
    print("="*60)