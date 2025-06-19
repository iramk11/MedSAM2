import torch
import os
from pathlib import Path

# Set environment variable to handle OpenMP error
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

def analyze_checkpoint(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        epoch = checkpoint.get('epoch', 'Unknown')
        train_loss = checkpoint.get('train_loss', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'Unknown')
        return {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
    except Exception as e:
        print(f"Error reading {checkpoint_path}: {e}")
        return None

def main():
    checkpoint_dir = Path('/Users/iram.kamdar/Desktop/medsam_training/MedSAM2/checkpoints')
    checkpoint_files = [
        'medsam2_finetune_epoch_20250611_090557_5.pt',
        'medsam2_finetune_epoch_20250611_090557_10.pt',
        'medsam2_finetune_epoch_20250611_090557_15.pt',
        'medsam2_finetune_epoch_20250611_090557_20.pt',
        'medsam2_finetune_epoch_20250611_090557_25.pt',
        'medsam2_finetune_BEST_20250611_090557.pt'
    ]

    results = []
    for file in checkpoint_files:
        path = checkpoint_dir / file
        if path.exists():
            result = analyze_checkpoint(path)
            if result:
                results.append(result)
                print(f"\nFile: {file}")
                print(f"Epoch: {result['epoch']}")
                print(f"Train Loss: {result['train_loss']}")
                print(f"Val Loss: {result['val_loss']}")

    if results:
        val_losses = [r['val_loss'] for r in results if r['val_loss'] != 'Unknown']
        if val_losses:
            print("\nValidation Loss Statistics:")
            print(f"Minimum Val Loss: {min(val_losses)}")
            print(f"Maximum Val Loss: {max(val_losses)}")
            print(f"Average Val Loss: {sum(val_losses)/len(val_losses)}")

if __name__ == "__main__":
    main() 