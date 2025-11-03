import os
import pandas as pd
import requests
from PIL import Image
import zipfile
from tqdm import tqdm
import shutil
from pathlib import Path

class Flickr8kDatasetSetup:
    """
    Complete setup for Flickr8k dataset
    Note: You'll need to download the dataset from:
    https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
    https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip
    """
    
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.images_dir = self.processed_dir / "images"
        self.sample_dir = self.processed_dir / "samples"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.images_dir, self.sample_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self):
        """
        Manual download instructions for Flickr8k
        """
        print("=" * 60)
        print("FLICKR8K DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        print("\n1. Download the following files:")
        print("   - Images: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip")
        print("   - Captions: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip")
        print("\n2. Place both zip files in the 'data/raw/' directory")
        print("\n3. Run the extract_and_process() method")
        print("=" * 60)
        
        # Alternative: Using Kaggle API (if you have it set up)
        print("\nAlternative - Using Kaggle CLI:")
        print("kaggle datasets download -d adityajn105/flickr8k -p data/raw/")
        
    def extract_and_process(self):
        """Extract zip files and organize data"""
        # Extract images
        image_zip = self.raw_dir / "Flickr8k_Dataset.zip"
        text_zip = self.raw_dir / "Flickr8k_text.zip"
        
        if image_zip.exists():
            print("Extracting images...")
            with zipfile.ZipFile(image_zip, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)
        
        if text_zip.exists():
            print("Extracting captions...")
            with zipfile.ZipFile(text_zip, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)
        
        print("Files extracted successfully!")
    
    def load_captions(self):
        """Load and process captions from Flickr8k_text"""
        captions_file = self.raw_dir / "Flickr8k.token.txt"
        
        if not captions_file.exists():
            captions_file = self.raw_dir / "Flickr8k_text" / "Flickr8k.token.txt"
        
        captions_data = []
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Format: image_id#caption_num\tcaption_text
                    parts = line.split('\t')
                    if len(parts) == 2:
                        img_caption_id, caption = parts
                        img_id = img_caption_id.split('#')[0]
                        caption_num = img_caption_id.split('#')[1]
                        
                        captions_data.append({
                            'image_id': img_id,
                            'caption_num': int(caption_num),
                            'caption': caption.strip()
                        })
        
        df = pd.DataFrame(captions_data)
        print(f"Loaded {len(df)} captions for {df['image_id'].nunique()} images")
        
        # Save processed captions
        df.to_csv(self.processed_dir / "captions.csv", index=False)
        
        return df
    
    def clean_captions(self, df):
        """Clean and preprocess captions"""
        import re
        import string
        
        def clean_text(text):
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation except spaces
            text = re.sub(f'[{string.punctuation}]', '', text)
            # Remove extra whitespaces
            text = ' '.join(text.split())
            # Remove trailing spaces
            text = text.strip()
            return text
        
        print("Cleaning captions...")
        df['caption_clean'] = df['caption'].apply(clean_text)
        
        # Save cleaned captions
        df.to_csv(self.processed_dir / "captions_clean.csv", index=False)
        
        # Create a simplified version with one row per image (using first caption)
        df_simple = df[df['caption_num'] == 0][['image_id', 'caption_clean']].copy()
        df_simple.to_csv(self.processed_dir / "captions_simple.csv", index=False)
        
        print(f"Cleaned {len(df)} captions")
        return df
    
    def organize_images(self, sample_size=1000):
        """Copy and organize images, create sample subset"""
        # Source image directory
        source_imgs = self.raw_dir / "Flicker8k_Dataset"
        if not source_imgs.exists():
            source_imgs = self.raw_dir / "Flickr8k_Dataset"
        
        if not source_imgs.exists():
            print(f"Error: Image directory not found at {source_imgs}")
            return
        
        # Load captions to get image list
        captions_df = pd.read_csv(self.processed_dir / "captions_clean.csv")
        unique_images = captions_df['image_id'].unique()[:sample_size]
        
        print(f"Copying {len(unique_images)} images to processed directory...")
        
        for img_id in tqdm(unique_images):
            source_path = source_imgs / img_id
            if source_path.exists():
                # Copy to processed directory
                dest_path = self.images_dir / img_id
                shutil.copy2(source_path, dest_path)
                
                # Also copy first sample_size images to sample directory
                if len(list(self.sample_dir.glob("*.jpg"))) < 100:  # Keep sample dir small
                    shutil.copy2(source_path, self.sample_dir / img_id)
        
        print(f"Images organized! Full set: {len(list(self.images_dir.glob('*.jpg')))} images")
        print(f"Sample set: {len(list(self.sample_dir.glob('*.jpg')))} images")
    
    def create_train_val_test_split(self):
        """Create train/val/test splits"""
        captions_df = pd.read_csv(self.processed_dir / "captions_clean.csv")
        
        # Official Flickr8k splits
        train_file = self.raw_dir / "Flickr8k_text" / "Flickr_8k.trainImages.txt"
        val_file = self.raw_dir / "Flickr8k_text" / "Flickr_8k.devImages.txt"
        test_file = self.raw_dir / "Flickr8k_text" / "Flickr_8k.testImages.txt"
        
        if train_file.exists():
            # Use official splits
            with open(train_file) as f:
                train_imgs = set(line.strip() for line in f)
            with open(val_file) as f:
                val_imgs = set(line.strip() for line in f)
            with open(test_file) as f:
                test_imgs = set(line.strip() for line in f)
        else:
            # Create custom 70-15-15 split
            unique_imgs = captions_df['image_id'].unique()
            n_imgs = len(unique_imgs)
            
            train_size = int(0.7 * n_imgs)
            val_size = int(0.15 * n_imgs)
            
            train_imgs = set(unique_imgs[:train_size])
            val_imgs = set(unique_imgs[train_size:train_size + val_size])
            test_imgs = set(unique_imgs[train_size + val_size:])
        
        # Add split column to dataframe
        captions_df['split'] = captions_df['image_id'].apply(
            lambda x: 'train' if x in train_imgs else ('val' if x in val_imgs else 'test')
        )
        
        # Save split information
        captions_df.to_csv(self.processed_dir / "captions_with_splits.csv", index=False)
        
        print(f"Dataset split:")
        print(f"  Train: {len(train_imgs)} images")
        print(f"  Val: {len(val_imgs)} images")
        print(f"  Test: {len(test_imgs)} images")
        
        return captions_df
    
    def get_stats(self):
        """Print dataset statistics"""
        captions_df = pd.read_csv(self.processed_dir / "captions_clean.csv")
        
        print("\n" + "=" * 60)
        print("FLICKR8K DATASET STATISTICS")
        print("=" * 60)
        print(f"Total images: {captions_df['image_id'].nunique()}")
        print(f"Total captions: {len(captions_df)}")
        print(f"Captions per image: {len(captions_df) / captions_df['image_id'].nunique():.1f}")
        
        # Caption length statistics
        captions_df['caption_length'] = captions_df['caption_clean'].str.split().str.len()
        print(f"\nCaption lengths:")
        print(f"  Min: {captions_df['caption_length'].min()} words")
        print(f"  Max: {captions_df['caption_length'].max()} words")
        print(f"  Avg: {captions_df['caption_length'].mean():.1f} words")
        print(f"  Median: {captions_df['caption_length'].median():.1f} words")
        
        # Most common words
        from collections import Counter
        all_words = ' '.join(captions_df['caption_clean']).split()
        word_freq = Counter(all_words)
        
        print(f"\nTop 10 most common words:")
        for word, count in word_freq.most_common(10):
            print(f"  {word}: {count}")
        
        print("=" * 60)


# Main setup script
if __name__ == "__main__":
    print("Starting Flickr8k Dataset Setup...")
    
    # Initialize setup
    setup = Flickr8kDatasetSetup(base_dir="data")
    
    # Step 1: Download instructions
    setup.download_dataset()
    
    # Step 2: Extract (run after downloading)
    setup.extract_and_process()
    
    # Step 3: Load and clean captions
    df_captions = setup.load_captions()
    df_clean = setup.clean_captions(df_captions)
    
    # Step 4: Organize images (use sample_size=1000 for milestone 1)
    setup.organize_images(sample_size=1000)
    
    # Step 5: Create train/val/test splits
    setup.create_train_val_test_split()
    
    # Step 6: Show statistics
    setup.get_stats()
    
    print("\nUncomment the steps above after downloading the dataset!")