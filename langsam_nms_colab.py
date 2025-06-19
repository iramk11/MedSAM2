#!/usr/bin/env python3
"""
JUPYTER NOTEBOOK VERSION - Text prompt-based polyp detection using LangSAM with NMS.
Copy this entire code into ONE Jupyter notebook cell and run it.
ENHANCED VERSION - Includes Non-Max Suppression for better detection filtering.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from PIL import Image as PILImage
import torch
from torchvision.ops import nms
import warnings
warnings.filterwarnings('ignore')

# ================================
# 🔧 CONFIGURATION - EDIT THESE!
# ================================

# Multiple text prompts to search for (will try ALL of these)
TEXT_PROMPTS = [
    "highlight the raised circular lesion (polyp) inside the colon"
]

# Input images (choose ONE option)
# Option 1: Single image - SET TO None FOR BATCH PROCESSING
SINGLE_IMAGE = None  # Set to None for batch processing

# Option 2: Batch processing 
IMAGES_DIR = "/content/drive/MyDrive/val_images"
MAX_FILES = 10  # Process first 10 images for testing (set to None for all)

# Output directory
OUTPUT_DIR = "/content/drive/MyDrive/text_prompt_results3"

# Detection thresholds (0-1, lower = more sensitive)
BOX_THRESHOLD = 0.15    # Object detection confidence (lowered for multi-prompt)
TEXT_THRESHOLD = 0.15   # Text-object association confidence (lowered for multi-prompt)
NMS_IOU_THRESHOLD = 0.4 # Non-Max Suppression IoU threshold

# Try to import LangSAM
try:
    from samgeo.text_sam import LangSAM
    LANGSAM_AVAILABLE = True
    print("✅ LangSAM is available!")
except ImportError:
    LANGSAM_AVAILABLE = False
    print("❌ LangSAM not available!")
    print("📥 Install with: !pip install segment-geospatial groundingdino-py")
    print("⚠️  Then restart your kernel and run this cell again.")

if LANGSAM_AVAILABLE:
    
    # ================================
    # 🚀 MAIN CODE
    # ================================
    
    print("\n" + "="*60)
    print("🏥 MEDICAL TEXT PROMPT SEGMENTATION + NMS")
    print(f"🔤 Text Prompts: {len(TEXT_PROMPTS)} prompts")
    for i, prompt in enumerate(TEXT_PROMPTS, 1):
        print(f"   {i}. '{prompt}'")
    print("🤖 Technology: LangSAM (GroundingDINO + SAM) + Non-Max Suppression")
    print(f"🎯 NMS IoU Threshold: {NMS_IOU_THRESHOLD}")
    print("="*60)
    
    # Initialize LangSAM with better error handling
    print("\n🤖 Initializing LangSAM (this may take a few minutes)...")
    print("📥 Downloading model weights if not already cached...")
    
    try:
        # Try with a more stable model type first
        print("🔧 Trying sam-vit-base model...")
        sam = LangSAM(model_type="vit_b")
        print("✅ LangSAM initialized successfully with vit_b model!")
        
        # Create output directory
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
        
        def apply_nms_to_detections(boxes, scores=None, iou_threshold=0.4):
            """Apply Non-Max Suppression to filter overlapping detections."""
            if boxes is None or len(boxes) <= 1:
                return boxes, scores
            
            try:
                # Convert to torch tensors if not already
                if not isinstance(boxes, torch.Tensor):
                    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                else:
                    boxes_tensor = boxes.clone()
                
                # Handle scores
                if scores is not None:
                    if not isinstance(scores, torch.Tensor):
                        scores_tensor = torch.tensor(scores, dtype=torch.float32)
                    else:
                        scores_tensor = scores.clone()
                else:
                    # If no scores available, use dummy scores (all equal)
                    scores_tensor = torch.ones(len(boxes_tensor), dtype=torch.float32)
                
                # Apply NMS
                keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=iou_threshold)
                
                # Filter results
                filtered_boxes = boxes_tensor[keep_indices].numpy()
                filtered_scores = scores_tensor[keep_indices].numpy() if scores is not None else None
                
                return filtered_boxes, filtered_scores, len(boxes) - len(filtered_boxes)
                
            except Exception as e:
                print(f"      ⚠️  NMS failed: {e}, using original detections")
                return boxes, scores, 0
        
        def process_single_image_multi_prompts(image_path, text_prompts, sam_model):
            """Process a single image with multiple text prompts and combine results."""
            
            filename = Path(image_path).stem
            print(f"\n📸 Processing: {Path(image_path).name}")
            print(f"🔍 Testing {len(text_prompts)} text prompts...")
            
            try:
                # Load and prepare image
                image = PILImage.open(image_path).convert('RGB')
                
                # Save temporary image for LangSAM
                temp_image_path = output_dir / f"{filename}_temp.jpg"
                image.save(temp_image_path)
                
                # Store all results from different prompts
                all_results = []
                successful_prompts = []
                total_boxes_before_nms = 0
                total_boxes_after_nms = 0
                
                # Run text-prompted segmentation for each prompt
                for i, text_prompt in enumerate(text_prompts, 1):
                    try:
                        print(f"   🧠 Prompt {i}/{len(text_prompts)}: '{text_prompt[:50]}{'...' if len(text_prompt) > 50 else ''}'")
                        
                        sam_model.predict(
                            str(temp_image_path), 
                            text_prompt, 
                            box_threshold=BOX_THRESHOLD,
                            text_threshold=TEXT_THRESHOLD
                        )
                        
                        # Check if any detections were found
                        if hasattr(sam_model, 'boxes') and len(sam_model.boxes) > 0:
                            original_boxes = sam_model.boxes
                            original_scores = sam_model.scores if hasattr(sam_model, 'scores') else None
                            
                            # Apply NMS to filter overlapping detections
                            filtered_boxes, filtered_scores, removed_count = apply_nms_to_detections(
                                original_boxes, 
                                original_scores, 
                                NMS_IOU_THRESHOLD
                            )
                            
                            if len(filtered_boxes) > 0:
                                print(f"      🔍 NMS: {len(original_boxes)} → {len(filtered_boxes)} boxes (removed {removed_count} overlapping)")
                                
                                # Update the model's attributes with filtered results
                                sam_model.boxes = filtered_boxes
                                if hasattr(sam_model, 'scores') and filtered_scores is not None:
                                    sam_model.scores = filtered_scores
                                
                                all_results.append({
                                    'prompt': text_prompt,
                                    'boxes': filtered_boxes,
                                    'original_boxes': original_boxes,
                                    'boxes_removed': removed_count,
                                    'masks': sam_model.masks if hasattr(sam_model, 'masks') else None,
                                    'phrases': sam_model.phrases if hasattr(sam_model, 'phrases') else None
                                })
                                successful_prompts.append(text_prompt)
                                total_boxes_before_nms += len(original_boxes)
                                total_boxes_after_nms += len(filtered_boxes)
                                print(f"      ✅ Found {len(filtered_boxes)} detection(s) after NMS")
                            else:
                                print(f"      ⚠️  All boxes removed by NMS")
                        else:
                            print(f"      ❌ No detections found")
                            
                    except Exception as e:
                        print(f"      ❌ Error with prompt: {e}")
                        continue
                
                if not all_results:
                    print(f"❌ No polyps detected with any prompt for {Path(image_path).name}")
                    return False
                
                print(f"✅ Successfully detected polyps with {len(successful_prompts)}/{len(text_prompts)} prompts")
                print(f"🔍 NMS Summary: {total_boxes_before_nms} → {total_boxes_after_nms} total boxes (removed {total_boxes_before_nms - total_boxes_after_nms} overlapping)")
                
                # Use the best result (most detections) for final output
                best_result = max(all_results, key=lambda x: len(x['boxes']) if x['boxes'] is not None else 0)
                best_prompt = best_result['prompt']
                
                print(f"🏆 Best result from: '{best_prompt[:50]}{'...' if len(best_prompt) > 50 else ''}' ({len(best_result['boxes'])} detections)")
                
                # Generate final outputs using the best result
                # Re-run prediction with best prompt for clean output generation
                sam_model.predict(
                    str(temp_image_path), 
                    best_prompt, 
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD
                )
                
                # Apply NMS to final result as well
                if hasattr(sam_model, 'boxes') and len(sam_model.boxes) > 0:
                    filtered_boxes, filtered_scores, _ = apply_nms_to_detections(
                        sam_model.boxes, 
                        sam_model.scores if hasattr(sam_model, 'scores') else None,
                        NMS_IOU_THRESHOLD
                    )
                    sam_model.boxes = filtered_boxes
                    if hasattr(sam_model, 'scores') and filtered_scores is not None:
                        sam_model.scores = filtered_scores
                    
                    # Debug: Show what we have after NMS
                    print(f"   🔍 Debug: Final boxes shape: {filtered_boxes.shape if hasattr(filtered_boxes, 'shape') else len(filtered_boxes)}")
                    print(f"   🔍 Debug: Box coordinates: {filtered_boxes[:3] if len(filtered_boxes) > 0 else 'None'}")  # Show first 3 boxes
                    if hasattr(sam_model, 'scores') and sam_model.scores is not None:
                        print(f"   🔍 Debug: Scores: {sam_model.scores[:3] if len(sam_model.scores) > 0 else 'None'}")  # Show first 3 scores
                
                # Save results with better error handling for data types
                try:
                    # Create custom visualization with explicit bounding boxes
                    bbox_output = output_dir / f"{filename}_with_boxes.png"
                    
                    # Load original image - use original path, not temp
                    img_with_boxes = cv2.imread(str(image_path))
                    if img_with_boxes is None:
                        # Fallback to numpy array from PIL
                        img_with_boxes = np.array(image)
                    else:
                        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                    
                    # Draw bounding boxes manually
                    if hasattr(sam_model, 'boxes') and sam_model.boxes is not None and len(sam_model.boxes) > 0:
                        for i, box in enumerate(sam_model.boxes):
                            x1, y1, x2, y2 = [int(coord) for coord in box]
                            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue boxes
                            
                            # Add confidence score if available
                            if hasattr(sam_model, 'scores') and sam_model.scores is not None:
                                score = sam_model.scores[i]
                                cv2.putText(img_with_boxes, f'{score:.2f}', (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Save using matplotlib for better control
                    plt.figure(figsize=(12, 8))
                    plt.imshow(img_with_boxes)
                    plt.title(f"Multi-Prompt Detection + NMS ({len(successful_prompts)} prompts successful)\n{len(sam_model.boxes) if hasattr(sam_model, 'boxes') and sam_model.boxes is not None else 0} boxes after NMS")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(bbox_output, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"   📦 Saved with explicit boxes: {bbox_output}")
                    
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not save bbox image ({e})")
                    # Fallback to original method
                    try:
                        bbox_output = output_dir / f"{filename}_with_boxes_fallback.png"
                        sam_model.show_anns(
                            cmap="Blues",
                            box_color="red",
                            title=f"Multi-Prompt Detection + NMS ({len(successful_prompts)} prompts successful)",
                            blend=True,
                            output=str(bbox_output)
                        )
                        print(f"   📦 Saved (fallback): {bbox_output}")
                    except:
                        bbox_output = None
                
                try:
                    # Save results without bounding boxes
                    mask_output = output_dir / f"{filename}_mask_only.png"
                    sam_model.show_anns(
                        cmap="Blues",
                        add_boxes=False,
                        alpha=0.7,
                        title=f"Best Prompt + NMS: {best_prompt[:40]}...",
                        output=str(mask_output)
                    )
                    print(f"   🎭 Saved: {mask_output}")
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not save mask image ({e})")
                    mask_output = None
                
                try:
                    # Save binary mask with better error handling
                    binary_output = output_dir / f"{filename}_binary.png"
                    
                    # Try to create binary mask manually if show_anns fails
                    try:
                        sam_model.show_anns(
                            cmap="Greys_r",
                            add_boxes=False,
                            alpha=1,
                            title=f"Binary Mask + NMS ({len(best_result['boxes'])} detections)",
                            blend=False,
                            output=str(binary_output)
                        )
                        print(f"   ⚫ Saved: {binary_output}")
                    except Exception as mask_error:
                        print(f"   ⚠️  LangSAM binary save failed ({mask_error}), trying manual approach...")
                        
                        # Manual binary mask creation
                        if hasattr(sam_model, 'masks') and sam_model.masks is not None:
                            # Convert masks to proper format
                            combined_mask = np.zeros((image.height, image.width), dtype=np.uint8)
                            for mask in sam_model.masks:
                                if hasattr(mask, 'shape'):
                                    mask_array = np.array(mask, dtype=np.uint8)
                                    if mask_array.shape[:2] == combined_mask.shape:
                                        combined_mask = np.logical_or(combined_mask, mask_array).astype(np.uint8)
                            
                            # Save manually created binary mask
                            plt.figure(figsize=(10, 8))
                            plt.imshow(combined_mask * 255, cmap='gray')
                            plt.title(f"Binary Mask + NMS ({len(best_result['boxes'])} detections)")
                            plt.axis('off')
                            plt.tight_layout()
                            plt.savefig(binary_output, dpi=150, bbox_inches='tight')
                            plt.close()
                            print(f"   ⚫ Saved (manual): {binary_output}")
                        else:
                            print(f"   ⚠️  No masks available for binary image")
                            binary_output = None
                            
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not save binary image ({e})")
                    binary_output = None
                
                # Save prompt analysis results
                analysis_file = output_dir / f"{filename}_prompt_analysis.txt"
                with open(analysis_file, 'w') as f:
                    f.write(f"Multi-Prompt Analysis + NMS for {filename}\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Architecture: LangSAM + Non-Max Suppression\n")
                    f.write(f"NMS IoU threshold: {NMS_IOU_THRESHOLD}\n")
                    f.write(f"Total prompts tested: {len(text_prompts)}\n")
                    f.write(f"Successful prompts: {len(successful_prompts)}\n")
                    f.write(f"Best prompt: {best_prompt}\n")
                    f.write(f"Best prompt detections: {len(best_result['boxes'])}\n")
                    f.write(f"Total boxes before NMS: {total_boxes_before_nms}\n")
                    f.write(f"Total boxes after NMS: {total_boxes_after_nms}\n")
                    f.write(f"Boxes removed by NMS: {total_boxes_before_nms - total_boxes_after_nms}\n\n")
                    f.write("Prompt Results:\n")
                    for result in all_results:
                        f.write(f"- '{result['prompt']}': {len(result['original_boxes'])} → {len(result['boxes'])} detections (removed {result['boxes_removed']})\n")
                
                # Clean up temp file at the very end
                try:
                    if temp_image_path.exists():
                        temp_image_path.unlink()
                        print(f"   🗑️  Cleaned up temp file")
                except Exception as cleanup_error:
                    print(f"   ⚠️  Could not clean up temp file: {cleanup_error}")
                
                print(f"💾 Results summary:")
                if bbox_output and bbox_output.exists():
                    print(f"   📦 With boxes: {bbox_output}")
                if mask_output and mask_output.exists():
                    print(f"   🎭 Mask only: {mask_output}")
                if binary_output and binary_output.exists():
                    print(f"   ⚫ Binary: {binary_output}")
                print(f"   📊 Analysis: {analysis_file}")
                
                # Display results in notebook only if we have images to show
                try:
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    
                    # Original
                    axes[0].imshow(image)
                    axes[0].set_title('Original')
                    axes[0].axis('off')
                    
                    # Create explicit bounding box visualization for display
                    if hasattr(sam_model, 'boxes') and sam_model.boxes is not None and len(sam_model.boxes) > 0:
                        # Load image and draw boxes - use original image instead of temp
                        img_with_boxes = cv2.imread(str(image_path))
                        if img_with_boxes is None:
                            # Fallback to numpy array from PIL
                            img_with_boxes = np.array(image)
                        else:
                            img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                        
                        # Draw each bounding box
                        for i, box in enumerate(sam_model.boxes):
                            x1, y1, x2, y2 = [int(coord) for coord in box]
                            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue boxes
                            
                            # Add confidence score if available
                            if hasattr(sam_model, 'scores') and sam_model.scores is not None:
                                score = sam_model.scores[i]
                                cv2.putText(img_with_boxes, f'{score:.2f}', (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        axes[1].imshow(img_with_boxes)
                        axes[1].set_title(f'With Boxes + NMS ({len(sam_model.boxes)} boxes)')
                        axes[1].axis('off')
                    else:
                        axes[1].text(0.5, 0.5, 'No Boxes\nDetected', ha='center', va='center', transform=axes[1].transAxes)
                        axes[1].set_title('With Boxes + NMS')
                        axes[1].axis('off')
                    
                    if mask_output and mask_output.exists():
                        mask_img = PILImage.open(mask_output)
                        axes[2].imshow(mask_img)
                        axes[2].set_title('Mask Only + NMS')
                        axes[2].axis('off')
                    else:
                        axes[2].text(0.5, 0.5, 'Not\nGenerated', ha='center', va='center', transform=axes[2].transAxes)
                        axes[2].set_title('Mask Only + NMS')
                        axes[2].axis('off')
                    
                    if binary_output and binary_output.exists():
                        binary_img = PILImage.open(binary_output)
                        axes[3].imshow(binary_img, cmap='gray')
                        axes[3].set_title('Binary Mask + NMS')
                        axes[3].axis('off')
                    else:
                        axes[3].text(0.5, 0.5, 'Not\nGenerated', ha='center', va='center', transform=axes[3].transAxes)
                        axes[3].set_title('Binary Mask + NMS')
                        axes[3].axis('off')
                    
                    plt.suptitle(f"Multi-Prompt + NMS Results: {len(successful_prompts)}/{len(text_prompts)} prompts successful", fontsize=14)
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"   ⚠️  Could not display results in notebook: {e}")
                
                return True
                
            except Exception as e:
                print(f"❌ Error processing {Path(image_path).name}: {e}")
                return False
        
        # Determine processing mode
        print(f"\n🔧 Configuration:")
        print(f"   Single Image: {SINGLE_IMAGE}")
        print(f"   Images Dir: {IMAGES_DIR}")
        print(f"   Max Files: {MAX_FILES}")
        print(f"   Text Prompts: {len(TEXT_PROMPTS)}")
        print(f"   NMS IoU Threshold: {NMS_IOU_THRESHOLD}")
        
        if SINGLE_IMAGE and os.path.exists(SINGLE_IMAGE):
            # Process single image
            print(f"\n🖼️  Processing single image: {SINGLE_IMAGE}")
            success = process_single_image_multi_prompts(SINGLE_IMAGE, TEXT_PROMPTS, sam)
            
            if success:
                print("\n🎉 Processing completed successfully!")
            else:
                print("\n❌ Processing failed!")
                
        else:
            # Process batch of images
            print(f"\n🎯 BATCH PROCESSING MODE")
            
            if not os.path.exists(IMAGES_DIR):
                print(f"❌ Images directory not found: {IMAGES_DIR}")
            else:
                images_dir = Path(IMAGES_DIR)
                jpg_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
                
                print(f"📁 Found {len(jpg_files)} total JPG files")
                
                if MAX_FILES and MAX_FILES > 0:
                    jpg_files = jpg_files[:MAX_FILES]
                    print(f"🔢 Limited to first {len(jpg_files)} files")
                
                if not jpg_files:
                    print(f"❌ No JPG images found in {IMAGES_DIR}")
                else:
                    print(f"\n🎯 Processing {len(jpg_files)} images with {len(TEXT_PROMPTS)} text prompts each + NMS")
                    
                    successful = 0
                    failed = 0
                    total_removed_by_nms = 0
                    
                    for i, image_path in enumerate(jpg_files, 1):
                        print(f"\n{'='*50}")
                        print(f"📷 Processing {i}/{len(jpg_files)}: {image_path.name}")
                        
                        if process_single_image_multi_prompts(image_path, TEXT_PROMPTS, sam):
                            successful += 1
                        else:
                            failed += 1
                    
                    print(f"\n{'='*50}")
                    print(f"📊 BATCH PROCESSING SUMMARY:")
                    print(f"✅ Successful: {successful}")
                    print(f"❌ Failed: {failed}")
                    print(f"📁 Results saved to: {output_dir}")
                    print(f"🔍 Total prompt combinations tested: {len(jpg_files)} × {len(TEXT_PROMPTS)} = {len(jpg_files) * len(TEXT_PROMPTS)}")
                    print(f"🏗️  Architecture: LangSAM (GroundingDINO + SAM) + NMS (IoU={NMS_IOU_THRESHOLD})")
        
    except Exception as e:
        print(f"❌ Error initializing LangSAM: {e}")
        print("💡 This could be due to:")
        print("   • Version compatibility issues")
        print("   • Missing dependencies")
        print("   • Model download failures")
        print("\n🔧 Try these fixes:")
        print("   1. Reinstall dependencies:")
        print("      !pip uninstall segment-geospatial groundingdino-py -y")
        print("      !pip install segment-geospatial==0.10.2")
        print("   2. Clear model cache and restart kernel")
        print("   3. Use different Python environment")

else:
    # Show installation instructions
    print("\n📥 INSTALLATION REQUIRED:")
    print("Run this in a notebook cell first:")
    print()
    print("!pip install segment-geospatial groundingdino-py")
    print()
    print("Then restart your kernel and run this cell again.")

print("\n🏁 Done!") 