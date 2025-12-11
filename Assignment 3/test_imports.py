"""
Test imports from Assignment 2 to Assignment 3
"""

import sys
import os
from pathlib import Path

print("="*60)
print("TESTING IMPORTS")
print("="*60)

# Show current directory
print(f"\nCurrent directory: {os.getcwd()}")

# Show directory structure
print("\nDirectory structure:")
for item in Path('.').iterdir():
    if item.is_dir():
        print(f"  📁 {item.name}/")
        if 'Assignment' in item.name or 'Assignemnt' in item.name:
            for subitem in item.iterdir():
                if subitem.suffix == '.py':
                    print(f"    📄 {subitem.name}")

# Test 1: Add Assignment 2 to path
print("\n--- Test 1: sys.path approach ---")
assignment2_path = str(Path('Assignemnt 2').resolve())
print(f"Adding to path: {assignment2_path}")
sys.path.insert(0, assignment2_path)

try:
    from train_diffusion import ConditionalUNet, DiffusionTrainer
    print("✓ Import successful!")
    print(f"  ConditionalUNet: {ConditionalUNet}")
    print(f"  DiffusionTrainer: {DiffusionTrainer}")
except Exception as e:
    print(f"✗ Import failed: {e}")

# Test 2: Check if file exists
print("\n--- Test 2: File existence ---")
train_file = Path('Assignemnt 2/train_diffusion.py')
print(f"File exists: {train_file.exists()}")
if train_file.exists():
    print(f"File size: {train_file.stat().st_size} bytes")

# Test 3: Try relative import
print("\n--- Test 3: Check sys.path ---")
print("Current sys.path entries:")
for p in sys.path[:5]:
    print(f"  {p}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)