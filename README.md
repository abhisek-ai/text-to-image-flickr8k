# Text-to-Image Generation with Flickr8k

## Project Overview

Implementation of a text-to-embedding pipeline using CLIP for the Flickr8k dataset. This project encodes natural language descriptions into embeddings for text-to-image generation tasks.

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/abhisek-ai/text-to-image-flickr8k.git
cd text-to-image-flickr8k

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Download Flickr8k dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and extract to `data/raw/`

## Usage

```bash
# 1. Preprocess dataset
python data.py

# 2. Generate embeddings
python baseline_pipeline.py

# 3. Test with 5 sample prompts
python test_five_prompts.py
```

## Project Structure

```
text-to-image-flickr8k/
├── data/                  # Dataset files
├── results/               # Output embeddings and visualizations
├── baseline_pipeline.py   # CLIP embedding pipeline
├── data.py               # Data preprocessing
├── test_five_prompts.py  # Test script
└── requirements.txt      # Dependencies
```

## Dataset Details

- **Flickr8k**: 8,091 images with 40,455 captions (5 per image)
- **Preprocessing**: Text normalization, lowercase, punctuation removal
- **Sample Size**: 1,000 image-caption pairs for testing

## Model

- **Text Encoder**: CLIP ViT-B/32 (OpenAI)
- **Embedding Dimension**: 512
- **Framework**: PyTorch with Hugging Face Transformers

## Results

Successfully tested 5 prompts:
1. "a dog playing in the park"
2. "sunset over the ocean with waves"
3. "a child eating ice cream happily"
4. "people walking in a busy street"
5. "mountains covered with snow"

All embeddings generated successfully with average processing time < 1 second per prompt.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- NumPy, Pandas, Pillow, Matplotlib

## Author

**Abhisek** - [GitHub](https://github.com/abhisek-ai)

## License

MIT License