# Text-to-Image Generation with Flickr8k

## Project Overview
Building a text-to-image generation model using CLIP embeddings and Flickr8k dataset.

## Setup Instructions

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset
Run the setup script to get download instructions:
```bash
python src/flickr8k_setup.py
```

### 3. Process Data
```bash
python src/flickr8k_setup.py --process
```

### 4. Run Baseline Pipeline
```bash
python src/baseline_pipeline.py
```

## Project Structure
```
text-to-image-flickr8k/
├── data/               # Dataset files
├── src/                # Source code
├── notebooks/          # Jupyter notebooks
├── results/            # Output results
├── docs/               # Documentation
└── configs/            # Configuration files
```

## Milestone 1 Progress
- [x] Dataset selection (Flickr8k)
- [x] Environment setup
- [ ] Data preprocessing
- [ ] Baseline pipeline
- [ ] 5 test prompts

## Author
[Your Name]
