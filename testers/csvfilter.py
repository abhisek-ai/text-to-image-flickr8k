import pandas as pd
from pathlib import Path

# Load CSV
df = pd.read_csv('data/processed/captions_clean.csv')
print(f"Total captions: {len(df)}")

# Get actual image files
img_dir = Path('Flicker8k_Dataset')
actual_images = set([f.name for f in img_dir.glob('*.jpg')])
print(f"Actual images: {len(actual_images)}")

# Filter CSV to only existing images
df_filtered = df[df['image_id'].isin(actual_images)]
print(f"Filtered captions: {len(df_filtered)}")

# Save
df_filtered.to_csv('data/processed/captions_8k_valid.csv', index=False)
print("Saved to captions_8k_valid.csv")