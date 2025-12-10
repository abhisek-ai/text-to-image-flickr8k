"""
Upscale 64x64 generated images for better presentation
Uses bicubic interpolation and optional sharpening
"""

from PIL import Image, ImageFilter, ImageEnhance
import argparse
from pathlib import Path

def upscale_image(input_path, output_path, scale_factor=4, sharpen=True, enhance=True):
    """
    Upscale image using high-quality interpolation
    
    Args:
        input_path: Path to 64x64 image
        output_path: Where to save upscaled image
        scale_factor: Multiplier (4 = 64x64 -> 256x256)
        sharpen: Apply sharpening filter
        enhance: Enhance contrast and color
    """
    print(f"Upscaling: {input_path}")
    
    # Load image
    img = Image.open(input_path)
    original_size = img.size
    print(f"  Original size: {original_size}")
    
    # Calculate new size
    new_size = (original_size[0] * scale_factor, original_size[1] * scale_factor)
    print(f"  Target size: {new_size}")
    
    # Method 1: Bicubic (smoothest)
    upscaled = img.resize(new_size, Image.BICUBIC)
    
    if sharpen:
        # Apply unsharp mask for clarity
        upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        print("  ✓ Sharpening applied")
    
    if enhance:
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(upscaled)
        upscaled = enhancer.enhance(1.2)
        
        # Enhance color saturation
        color_enhancer = ImageEnhance.Color(upscaled)
        upscaled = color_enhancer.enhance(1.15)
        print("  ✓ Enhancement applied")
    
    # Save
    upscaled.save(output_path, quality=95)
    print(f"  ✓ Saved to: {output_path}")
    print(f"  Final size: {upscaled.size}")
    
    return upscaled


def batch_upscale(input_dir, output_dir, scale_factor=4):
    """Upscale all images in directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    images = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg'))
    print(f"Found {len(images)} images to upscale\n")
    
    for img_path in images:
        output_file = output_path / f"{img_path.stem}_upscaled{img_path.suffix}"
        upscale_image(str(img_path), str(output_file), scale_factor)
        print()
    
    print(f"Upscaled {len(images)} images to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Upscale generated images')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, help='Output path (default: input_upscaled.png)')
    parser.add_argument('--scale', type=int, default=4, help='Scale factor (default: 4)')
    parser.add_argument('--no-sharpen', action='store_true', help='Disable sharpening')
    parser.add_argument('--no-enhance', action='store_true', help='Disable color/contrast enhancement')
    parser.add_argument('--batch', action='store_true', help='Process entire directory')
    
    args = parser.parse_args()
    
    if args.batch:
        output_dir = args.output or f"{args.input}_upscaled"
        batch_upscale(args.input, output_dir, args.scale)
    else:
        # Single image
        if not args.output:
            input_path = Path(args.input)
            args.output = f"{input_path.stem}_upscaled{input_path.suffix}"
        
        upscale_image(
            args.input, 
            args.output, 
            args.scale,
            sharpen=not args.no_sharpen,
            enhance=not args.no_enhance
        )


if __name__ == "__main__":
    print("""
Image Upscaler for Generated Images
====================================

Usage:

1. Upscale single image (64x64 -> 256x256):
   python upscale_image.py --input thisnew.png --output thisnew_big.png

2. Upscale to 512x512:
   python upscale_image.py --input thisnew.png --scale 8 --output thisnew_huge.png

3. Upscale without enhancements (just resize):
   python upscale_image.py --input thisnew.png --no-sharpen --no-enhance

4. Batch upscale entire folder:
   python upscale_image.py --input demo_images --batch --scale 4

Options:
  --scale 4     : 64x64 -> 256x256 (default)
  --scale 8     : 64x64 -> 512x512
  --scale 16    : 64x64 -> 1024x1024
  --no-sharpen  : Skip sharpening filter
  --no-enhance  : Skip contrast/color enhancement
    """)
    
    import sys
    if len(sys.argv) > 1:
        main()