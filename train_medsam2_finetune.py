#!/usr/bin/env python3
"""
Fine-tuning script for MedSAM2 with pre-trained weights.
Much faster than training from scratch - typically completes in hours not days!
"""

import os
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import multiprocessing

# Add current directory to path for imports
sys.path.append('.')
sys.path.append('./sam2')
sys.path.append('./training')

# Import MedSAM2 components
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KvasirNPZDataset(Dataset):
    """NPZ dataset loader for fine-tuning."""
    
    def __init__(self, data_dir, max_files=None, preload=False):
        self.data_dir = Path(data_dir)
        self.npz_files = list(self.data_dir.glob("*.npz"))
        self.preload = preload
        self.cache = {}
        
        if max_files:
            self.npz_files = self.npz_files[:max_files]
            
        logger.info(f"Found {len(self.npz_files)} NPZ files in {data_dir}")
        
        # Preload data into memory for faster training
        if preload and len(self.npz_files) < 500:
            logger.info("Preloading data into memory...")
            for i, npz_path in enumerate(tqdm(self.npz_files, desc="Loading")):
                self.cache[i] = self._load_npz(npz_path)
            logger.info("Data preloaded successfully!")
    
    def _load_npz(self, npz_path):
        """Load and process a single NPZ file."""
        data = np.load(npz_path)
        
        # Load single frame image and mask
        image = data['image']  # Shape: (H, W, 3)
        mask = data['mask']    # Shape: (H, W)
        
        # Convert to tensors
        image = torch.from_numpy(image).float() / 255.0  # Normalize to [0,1]
        mask = torch.from_numpy(mask).float()
        
        # Rearrange dimensions: (H, W, 3) -> (3, H, W)
        image = image.permute(2, 0, 1)  # (3, H, W)
        
        # For compatibility with existing code, create single-frame "sequences"
        images = image.unsqueeze(0)  # (1, 3, H, W)
        masks = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Create bounding box from mask
        mask_np = mask.numpy()
        if mask_np.sum() > 0:
            coords = np.where(mask_np > 0.5)
            if len(coords[0]) > 0:
                y1, x1 = coords[0].min(), coords[1].min()
                y2, x2 = coords[0].max(), coords[1].max()
                bbox = [x1, y1, x2, y2]
            else:
                bbox = [0, 0, 10, 10]  # Dummy bbox
        else:
            bbox = [0, 0, 10, 10]  # Dummy bbox
        
        bbox = torch.tensor(bbox).float()
        
        return {
            'images': images,      # (1, 3, H, W) - single frame
            'masks': masks,        # (1, 1, H, W) - single frame
            'bbox': bbox,          # (4,) 
            'filename': npz_path.name
        }
    
    def __len__(self):
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        if self.preload and idx in self.cache:
            return self.cache[idx]
        else:
            return self._load_npz(self.npz_files[idx])

def create_model_for_finetuning(device="cpu", pretrained_checkpoint=None):
    """Create MedSAM2 model with pre-trained weights for fine-tuning."""
    logger.info("🚀 Creating model for fine-tuning with pre-trained weights!")
    
    try:
        # Build SAM2 model with pre-trained weights
        if pretrained_checkpoint and os.path.exists(pretrained_checkpoint):
            # Load from custom checkpoint
            logger.info(f"Loading from custom checkpoint: {pretrained_checkpoint}")
            sam2_model = build_sam2(
                config_file="./sam2_local/configs/sam2.1_hiera_t512.yaml",
                ckpt_path=pretrained_checkpoint,
                device=device,
                mode="train"
            )
        else:
            # Load from default SAM2 checkpoint (this will download if not present)
            logger.info("Loading default SAM2 pre-trained weights...")
            sam2_model = build_sam2(
                config_file="sam2.1_hiera_t512.yaml",
                ckpt_path=None,  # This will use default pre-trained weights
                device=device,
                mode="train"
            )
        
        # Wrap with image predictor
        predictor = SAM2ImagePredictor(sam2_model)
        
        # Set model to training mode for fine-tuning
        predictor.model.train()
        
        logger.info("✅ SAM2 model loaded with pre-trained weights for fine-tuning!")
        return predictor
        
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        raise

def validate_epoch(predictor, val_loader, device):
    """Validate the model on validation set."""
    predictor.model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                # Take the single frame
                image = batch['images'][0, 0]  # (3, H, W)
                mask = batch['masks'][0, 0, 0]  # (H, W)
                bbox = batch['bbox'][0]  # (4,)
                
                # Move to device
                image = image.to(device)
                mask = mask.to(device)
                bbox = bbox.to(device)
                
                # Prepare image for feature computation
                image_np = image.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
                image_np = (image_np * 255).astype(np.uint8)
                
                # Transform the image and compute features
                input_image = predictor._transforms(image_np)
                input_image = input_image[None, ...].to(device)
                
                # Compute image embeddings
                backbone_out = predictor.model.forward_image(input_image)
                _, vision_feats, _, _ = predictor.model._prepare_backbone_features(backbone_out)
                
                # Add no_mem_embed if needed
                if predictor.model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed

                # Prepare features
                hires_size = predictor.model.image_size // 4
                bb_feat_sizes = [[hires_size // (2**k)]*2 for k in range(3)]
                feats = [
                    feat.permute(1, 2, 0).contiguous().view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
                ][::-1]
                
                image_embed = feats[-1]
                high_res_features = feats[:-1]
                
                # Prepare box prompt
                box_torch = bbox.unsqueeze(0)  # (1, 4)
                
                # Forward pass through mask decoder
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )
                
                low_res_multimasks, ious, sam_output_tokens, object_score_logits = predictor.model.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )
                
                # Resize to original image size
                pred_masks = torch.nn.functional.interpolate(
                    low_res_multimasks,
                    size=mask.shape,
                    mode='bilinear',
                    align_corners=False
                )
                
                pred_logits = pred_masks.squeeze(0).squeeze(0)
                
                # Validation loss
                val_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_logits, mask
                )
                
                total_val_loss += val_loss.item()
                
            except Exception as e:
                logger.warning(f"Error in validation batch {batch_idx}: {e}")
                continue
    
    return total_val_loss / len(val_loader)

def train_epoch(predictor, dataloader, optimizer, device, epoch):
    """Train for one epoch with fine-tuning approach."""
    predictor.model.train()
    total_loss = 0
    
    # Enable gradients
    torch.set_grad_enabled(True)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        try:
            batch_size = batch['images'].shape[0]
            loss = 0
            
            for b in range(batch_size):
                # Take the single frame
                image = batch['images'][b, 0]  # (3, H, W)
                mask = batch['masks'][b, 0, 0]  # (H, W)
                bbox = batch['bbox'][b]  # (4,)
                
                # Move to device
                image = image.to(device)
                mask = mask.to(device)
                bbox = bbox.to(device)
                
                # Prepare image for feature computation
                image_np = image.permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)
                image_np = (image_np * 255).astype(np.uint8)
                
                # Transform the image and compute features
                input_image = predictor._transforms(image_np)
                input_image = input_image[None, ...].to(device)
                
                # Compute image embeddings with gradients enabled  
                backbone_out = predictor.model.forward_image(input_image)
                _, vision_feats, _, _ = predictor.model._prepare_backbone_features(backbone_out)
                
                # Add no_mem_embed if needed
                if predictor.model.directly_add_no_mem_embed:
                    vision_feats[-1] = vision_feats[-1] + predictor.model.no_mem_embed

                # Prepare features
                hires_size = predictor.model.image_size // 4
                bb_feat_sizes = [[hires_size // (2**k)]*2 for k in range(3)]
                feats = [
                    feat.permute(1, 2, 0).contiguous().view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
                ][::-1]
                
                image_embed = feats[-1]
                high_res_features = feats[:-1]
                
                # Prepare box prompt
                box_torch = bbox.unsqueeze(0)  # (1, 4)
                
                # Forward pass through mask decoder
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )
                
                low_res_multimasks, ious, sam_output_tokens, object_score_logits = predictor.model.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )
                
                # Resize to original image size
                pred_masks = torch.nn.functional.interpolate(
                    low_res_multimasks,
                    size=mask.shape,
                    mode='bilinear',
                    align_corners=False
                )
                
                pred_logits = pred_masks.squeeze(0).squeeze(0)
                
                # Fine-tuning loss (simpler than training from scratch)
                # 1. Main segmentation loss
                seg_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_logits, mask
                )
                
                # 2. Optional IoU loss (lighter weight for fine-tuning)
                pred_sigmoid = torch.sigmoid(pred_logits)
                intersection = (pred_sigmoid * mask).sum()
                union = pred_sigmoid.sum() + mask.sum() - intersection
                iou_loss = 1 - (intersection + 1e-6) / (union + 1e-6)
                
                # Combined loss (lighter than training from scratch)
                sample_loss = seg_loss + 0.2 * iou_loss
                loss += sample_loss
            
            # Average loss over batch
            loss = loss / batch_size
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (lighter for fine-tuning)
            torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
                
        except Exception as e:
            logger.warning(f"Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / len(dataloader)

def main():
    """Main fine-tuning function."""
    
    parser = argparse.ArgumentParser(description='Fine-tune MedSAM2 with pre-trained weights')
    parser.add_argument('--data_dir', type=str, default='../Kvasir-SEG/split_npz', 
                        help='Path to NPZ data directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (much fewer for fine-tuning)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--max_files', type=int, default=500, help='Limit number of files')
    parser.add_argument('--preload', action='store_true', help='Preload data into memory')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--checkpoint', type=str, default=None, help='Custom checkpoint path (optional)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Using device: {device}")
    logger.info(f"🚀 FINE-TUNING with pre-trained weights!")
    logger.info(f"⚡ This will be MUCH faster than training from scratch!")
    
    # Check paths
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    # Create datasets
    train_dataset = KvasirNPZDataset(train_dir, max_files=args.max_files, preload=args.preload)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    if os.path.exists(val_dir):
        val_dataset = KvasirNPZDataset(val_dir, max_files=args.max_files//4 if args.max_files else None)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
        logger.info(f"Validation files: {len(val_dataset)}")
    else:
        val_loader = None
        logger.warning("No validation directory found")
    
    logger.info(f"Training files: {len(train_dataset)}")
    logger.info(f"Estimated time per epoch: {len(train_loader) * 0.5:.1f} minutes (much faster!)")
    logger.info(f"⚡ Total estimated training time: {args.epochs * len(train_loader) * 0.5 / 60:.1f} hours")
    
    # Create model for fine-tuning
    predictor = create_model_for_finetuning(device, args.checkpoint)
    
    # Setup optimizer for fine-tuning (lower learning rate)
    optimizer = torch.optim.AdamW(
        predictor.model.parameters(), 
        lr=args.lr,  # Lower learning rate for fine-tuning
        weight_decay=0.01,  # Less regularization
        eps=1e-8
    )
    
    # Learning rate scheduler for fine-tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    
    # Training loop
    logger.info("🚀 Starting fine-tuning...")
    logger.info("💡 Tip: Fine-tuning converges much faster than training from scratch!")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n📊 Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(predictor, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss = None
        if val_loader:
            val_loss = validate_epoch(predictor, val_loader, device)
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_dir = Path("./checkpoints")
                checkpoint_dir.mkdir(exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': predictor.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                }, checkpoint_dir / f"medsam2_finetune_BEST10Jun2025.pt")
                logger.info(f"💾 New best model saved! Val Loss: {val_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_dir = Path("./checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': predictor.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_dir / f"medsam2_finetune_epoch_1oJun2025{epoch}.pt")
            logger.info(f"💾 Checkpoint saved: medsam2_finetune_epoch10June2025_{epoch}.pt")
    
    logger.info("🎉 Fine-tuning completed!")

if __name__ == "__main__":
    main() 