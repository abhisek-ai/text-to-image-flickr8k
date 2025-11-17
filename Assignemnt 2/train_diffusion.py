"""
Conditional Diffusion Model Training on Flickr8k
Simple version for Assignment 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import CLIPTextModel, CLIPTokenizer
import os

# ==================== DATASET ====================
class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, captions_file, tokenizer, img_size=64):
        self.img_dir = Path(img_dir)
        self.tokenizer = tokenizer
        
        # Load captions
        import pandas as pd
        self.data = []
        df = pd.read_csv(captions_file)
        for _, row in df.iterrows():
            self.data.append({
                'image': row['image_id'],
                'caption': row['caption_clean']
            })
        
        print(f"Loaded {len(self.data)} image-caption pairs")
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = self.img_dir / item['image']
        
        if not img_path.exists():
            print(f"WARNING: Image not found: {img_path}")
            # Try next image instead of recursion
            idx = (idx + 1) % len(self.data)
            item = self.data[idx]
            img_path = self.img_dir / item['image']
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            raise  # Stop instead of infinite loop
        
        text_inputs = self.tokenizer(
            item['caption'],
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'image': image,
            'text_input_ids': text_inputs.input_ids.squeeze(0),
            'caption': item['caption']
    }


# ==================== MODEL ====================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalUNet(nn.Module):
    def __init__(self, img_channels=3, text_embed_dim=512, time_dim=256, base_ch=64):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, base_ch * 4),
            nn.GELU()
        )
        
        # Encoder
        self.enc1 = self._block(img_channels, base_ch)
        self.enc2 = self._block(base_ch, base_ch * 2)
        self.enc3 = self._block(base_ch * 2, base_ch * 4)
        
        # Bottleneck
        self.bottleneck = self._block(base_ch * 4, base_ch * 4)
        
        # Decoder
        self.dec3 = self._block(base_ch * 8, base_ch * 2)
        self.dec2 = self._block(base_ch * 4, base_ch)
        self.dec1 = self._block(base_ch * 2, base_ch)
        
        self.out = nn.Conv2d(base_ch, img_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.GELU()
        )
    
    def forward(self, x, t, text_embed):
        t_emb = self.time_mlp(t).view(t.shape[0], -1, 1, 1)
        text_emb = self.text_proj(text_embed).view(text_embed.shape[0], -1, 1, 1)
        
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        
        # Bottleneck with text conditioning
        bn = self.bottleneck(self.pool(x3))
        bn = bn + text_emb
        
        # Decoder
        d3 = self.dec3(torch.cat([self.up(bn), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), x1], dim=1))
        
        return self.out(d1)


# ==================== DIFFUSION ====================
class DiffusionTrainer:
    def __init__(self, model, text_encoder, timesteps=1000, device='cuda'):
        self.model = model
        self.text_encoder = text_encoder
        self.timesteps = timesteps
        self.device = device
        
        # Linear noise schedule
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None, None, None]
        
        return sqrt_alpha_prod * x0 + sqrt_one_minus_alpha_prod * noise
    
    def get_text_embeddings(self, input_ids):
        with torch.no_grad():
            outputs = self.text_encoder(input_ids)
            return outputs.pooler_output
    
    def train_step(self, batch, optimizer, cfg_dropout=0.1):
        images = batch['image'].to(self.device)
        text_input_ids = batch['text_input_ids'].to(self.device)
        batch_size = images.shape[0]
        
        # Random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # Add noise
        noise = torch.randn_like(images)
        noisy_images = self.add_noise(images, t, noise)
        
        # Get text embeddings with CFG dropout
        text_embeds = self.get_text_embeddings(text_input_ids)
        mask = torch.rand(batch_size, device=self.device) < cfg_dropout
        text_embeds = text_embeds * (~mask[:, None]).float()
        
        # Predict noise
        predicted_noise = self.model(noisy_images, t, text_embeds)
        loss = F.mse_loss(predicted_noise, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, text_input_ids, num_samples=1, img_size=64, guidance_scale=7.5):
        self.model.eval()
        
        text_embeds = self.get_text_embeddings(text_input_ids)
        text_embeds = text_embeds.repeat(num_samples, 1)
        uncond_embeds = torch.zeros_like(text_embeds)
        
        x = torch.randn(num_samples, 3, img_size, img_size, device=self.device)
        
        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling'):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
            
            # CFG
            noise_pred_cond = self.model(x, t, text_embeds)
            noise_pred_uncond = self.model(x, t, uncond_embeds)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Denoise
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred)
            x = x + torch.sqrt(beta) * noise
        
        self.model.train()
        return x


# ==================== TRAINING ====================
def train(img_dir, captions_file, epochs=5, batch_size=16, lr=1e-4, img_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CLIP
    print("Loading CLIP...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    # Dataset
    print("Loading dataset...")
    dataset = Flickr8kDataset(img_dir, captions_file, tokenizer, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Model
    print("Initializing model...")
    unet = ConditionalUNet().to(device)
    trainer = DiffusionTrainer(unet, text_encoder, device=device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    
    # Training
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    logs = []
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            loss = trainer.train_step(batch, optimizer)
            epoch_losses.append(loss)
            pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            if batch_idx % 50 == 0:
                logs.append({'epoch': epoch + 1, 'batch': batch_idx, 'loss': loss})
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoints/checkpoint_epoch_{epoch+1}.pt')
    
    # Save logs
    with open('logs/training_log.json', 'w') as f:
        json.dump(logs, f, indent=2)
    
    return trainer, tokenizer


# ==================== MAIN ====================
if __name__ == "__main__":
    # MODIFY THESE PATHS FOR YOUR SETUP
    IMG_DIR = "data/processed/samples"
    CAPTIONS_FILE = "data/processed/captions_filtered.csv"
    
    # Train
    trainer, tokenizer = train(
        img_dir=IMG_DIR,
        captions_file=CAPTIONS_FILE,
        epochs=5,
        batch_size=16,
        lr=1e-4,
        img_size=64
    )
    
    print("\nTraining complete! Checkpoints saved in ./checkpoints/")