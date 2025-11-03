# baseline_pipeline.py
"""
Baseline Text-to-Embedding Pipeline for Flickr8k
This creates a simple pipeline that:
1. Loads Flickr8k captions
2. Converts them to CLIP embeddings
3. Tests with 5 sample prompts
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# CLIP for embeddings
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel

# For visualization
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


class TextToEmbeddingPipeline:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initialize the text-to-embedding pipeline
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load CLIP model and processors
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_model = CLIPTextModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        
        # Store embeddings
        self.text_embeddings = []
        self.caption_texts = []
        
    def encode_text(self, text, return_pooled=True):
        """
        Convert text to CLIP embedding
        Args:
            text: Single text string or list of strings
            return_pooled: If True, return pooled output (512-dim), else return full sequence
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=77,  # CLIP's max length
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            if return_pooled:
                # Use pooled output for single vector representation
                embeddings = outputs.pooler_output
            else:
                # Use last hidden state for sequence representation
                embeddings = outputs.last_hidden_state
        
        return embeddings.cpu().numpy()
    
    def encode_image(self, image_path):
        """
        Convert image to CLIP embedding (for comparison)
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features.cpu().numpy()
    
    def process_flickr8k_captions(self, captions_file, sample_size=1000):
        """
        Process Flickr8k captions and create embeddings
        """
        print(f"Loading captions from {captions_file}")
        df = pd.read_csv(captions_file)
        
        # Sample if needed
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        print(f"Processing {len(df)} captions...")
        
        batch_size = 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(df), batch_size)):
            batch_texts = df['caption_clean'].iloc[i:i+batch_size].tolist()
            batch_embeddings = self.encode_text(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        self.text_embeddings = np.vstack(all_embeddings)
        self.caption_texts = df['caption_clean'].tolist()
        
        print(f"Created embeddings with shape: {self.text_embeddings.shape}")
        
        # Save embeddings
        np.save('data/processed/text_embeddings.npy', self.text_embeddings)
        
        return self.text_embeddings
    
    def find_similar_captions(self, query_text, top_k=5):
        """
        Find most similar captions to a query text
        """
        if len(self.text_embeddings) == 0:
            print("No embeddings loaded. Process captions first!")
            return []
        
        # Encode query
        query_embedding = self.encode_text(query_text)
        
        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.text_embeddings)[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'caption': self.caption_texts[idx],
                'similarity': similarities[idx]
            })
        
        return results
    
    def test_five_prompts(self):
        """
        Test with 5 diverse prompts as required for milestone
        """
        test_prompts = [
            "a dog playing in the park",
            "sunset over the ocean with waves",
            "a child eating ice cream",
            "people walking in a busy street",
            "mountains covered with snow"
        ]
        
        print("\n" + "="*60)
        print("TESTING 5 SAMPLE PROMPTS")
        print("="*60)
        
        results = {}
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n[Test {i}] Prompt: '{prompt}'")
            
            # Get embedding
            embedding = self.encode_text(prompt)
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
            
            # Find similar captions if we have processed them
            if len(self.caption_texts) > 0:
                similar = self.find_similar_captions(prompt, top_k=3)
                print(f"  Top 3 similar captions in dataset:")
                for j, item in enumerate(similar, 1):
                    print(f"    {j}. '{item['caption']}' (sim: {item['similarity']:.4f})")
            
            results[prompt] = {
                'embedding': embedding,
                'similar_captions': similar if len(self.caption_texts) > 0 else []
            }
        
        return results
    
    def visualize_embeddings(self, sample_size=100):
        """
        Visualize embeddings using t-SNE or PCA
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        if len(self.text_embeddings) == 0:
            print("No embeddings to visualize!")
            return
        
        # Sample embeddings for visualization
        sample_idx = np.random.choice(len(self.text_embeddings), 
                                     min(sample_size, len(self.text_embeddings)), 
                                     replace=False)
        sample_embeddings = self.text_embeddings[sample_idx]
        
        # Reduce dimensionality
        print("Running PCA for visualization...")
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(sample_embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        plt.title(f'CLIP Text Embeddings Visualization (n={len(sample_embeddings)})')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.tight_layout()
        plt.savefig('data/processed/embeddings_visualization.png', dpi=150)
        plt.show()
        
        print(f"Visualization saved to 'data/processed/embeddings_visualization.png'")
    
    def compute_dataset_stats(self):
        """
        Compute statistics about the embeddings
        """
        if len(self.text_embeddings) == 0:
            print("No embeddings loaded!")
            return
        
        print("\n" + "="*60)
        print("EMBEDDING STATISTICS")
        print("="*60)
        
        # Basic stats
        print(f"Number of embeddings: {len(self.text_embeddings)}")
        print(f"Embedding dimension: {self.text_embeddings.shape[1]}")
        print(f"Mean norm: {np.linalg.norm(self.text_embeddings, axis=1).mean():.4f}")
        print(f"Std norm: {np.linalg.norm(self.text_embeddings, axis=1).std():.4f}")
        
        # Compute pairwise similarities
        print("\nComputing pairwise similarities (this may take a moment)...")
        sample_size = min(500, len(self.text_embeddings))
        sample_idx = np.random.choice(len(self.text_embeddings), sample_size, replace=False)
        sample_embeddings = self.text_embeddings[sample_idx]
        
        similarities = cosine_similarity(sample_embeddings)
        
        # Remove diagonal (self-similarity)
        mask = ~np.eye(similarities.shape[0], dtype=bool)
        similarities_no_diag = similarities[mask]
        
        print(f"\nPairwise similarity statistics (n={sample_size} samples):")
        print(f"  Mean similarity: {similarities_no_diag.mean():.4f}")
        print(f"  Std similarity: {similarities_no_diag.std():.4f}")
        print(f"  Min similarity: {similarities_no_diag.min():.4f}")
        print(f"  Max similarity: {similarities_no_diag.max():.4f}")
        
        return {
            'n_embeddings': len(self.text_embeddings),
            'dimension': self.text_embeddings.shape[1],
            'mean_similarity': similarities_no_diag.mean(),
            'std_similarity': similarities_no_diag.std()
        }


# Main execution script
def main():
    """
    Main pipeline execution for milestone 1
    """
    print("="*60)
    print("TEXT-TO-EMBEDDING PIPELINE FOR FLICKR8K")
    print("="*60)
    
    # Initialize pipeline
    pipeline = TextToEmbeddingPipeline()
    
    # Path to processed captions
    captions_file = Path("data/processed/captions_clean.csv")
    
    if captions_file.exists():
        # Process Flickr8k captions
        pipeline.process_flickr8k_captions(captions_file, sample_size=1000)
        
        # Compute statistics
        pipeline.compute_dataset_stats()
        
        # Visualize embeddings
        pipeline.visualize_embeddings(sample_size=100)
    else:
        print(f"Warning: {captions_file} not found!")
        print("Please run the data setup script first.")
    
    # Test with 5 prompts (required for milestone)
    test_results = pipeline.test_five_prompts()
    
    # Save test results
    import pickle
    with open('data/processed/test_results.pkl', 'wb') as f:
        pickle.dump(test_results, f)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("Results saved to data/processed/")
    print("="*60)


if __name__ == "__main__":
    main()