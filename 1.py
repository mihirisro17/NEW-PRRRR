# Add this import at the top
from scipy.ndimage import binary_dilation

# ============= Buffer and Error Functions =============
def create_buffer(binary_array, buffer_pixels=2):
    """
    Expand binary mask by buffer_pixels on all sides using morphological dilation
    
    Args:
        binary_array: 2D numpy array with 0 and 1 values
        buffer_pixels: number of pixels to expand on each side
    
    Returns:
        buffered binary array
    """
    # Create a square structuring element
    # Size = 2*buffer_pixels + 1 to expand 'buffer_pixels' on each side
    struct_size = 2 * buffer_pixels + 1
    structuring_element = np.ones((struct_size, struct_size), dtype=bool)
    
    # Apply binary dilation
    buffered = binary_dilation(binary_array.astype(bool), structure=structuring_element)
    
    return buffered.astype(np.float32)


def compute_omission_commission_errors(hrdl, l1pm, buffer_pixels=2):
    """
    Compute omission and commission errors with buffer tolerance
    
    Args:
        hrdl: Ground truth binary mask (numpy array)
        l1pm: Predicted binary mask (numpy array)
        buffer_pixels: Buffer size in pixels
    
    Returns:
        Dictionary with error arrays and statistics
    """
    # Ensure binary
    hrdl = (hrdl > 0).astype(np.float32)
    l1pm = (l1pm > 0).astype(np.float32)
    
    # Create buffered versions
    hrdl_buffered = create_buffer(hrdl, buffer_pixels)
    l1pm_buffered = create_buffer(l1pm, buffer_pixels)
    
    # Omission Error: Ground truth pixels missed even with buffer tolerance
    # Pixels that are in HRDL but NOT in L1PM_buffered
    omission_error = np.logical_and(hrdl > 0, l1pm_buffered == 0).astype(np.float32)
    
    # Commission Error: Predicted pixels that are false positives even with buffer tolerance
    # Pixels that are in L1PM but NOT in HRDL_buffered
    commission_error = np.logical_and(l1pm > 0, hrdl_buffered == 0).astype(np.float32)
    
    # Calculate statistics
    total_hrdl_pixels = np.sum(hrdl)
    total_l1pm_pixels = np.sum(l1pm)
    
    omission_pixels = np.sum(omission_error)
    commission_pixels = np.sum(commission_error)
    
    # Error rates (as percentages)
    omission_rate = (omission_pixels / total_hrdl_pixels * 100) if total_hrdl_pixels > 0 else 0
    commission_rate = (commission_pixels / total_l1pm_pixels * 100) if total_l1pm_pixels > 0 else 0
    
    # Overall accuracy considering buffers
    # True positives: HRDL pixels covered by L1PM_buffered
    tp_buffered = np.sum(np.logical_and(hrdl > 0, l1pm_buffered > 0))
    
    results = {
        'hrdl': hrdl,
        'l1pm': l1pm,
        'hrdl_buffered': hrdl_buffered,
        'l1pm_buffered': l1pm_buffered,
        'omission_error': omission_error,
        'commission_error': commission_error,
        'stats': {
            'total_hrdl_pixels': int(total_hrdl_pixels),
            'total_l1pm_pixels': int(total_l1pm_pixels),
            'omission_pixels': int(omission_pixels),
            'commission_pixels': int(commission_pixels),
            'omission_rate_percent': omission_rate,
            'commission_rate_percent': commission_rate,
            'buffer_pixels': buffer_pixels
        }
    }
    
    return results


def print_error_report(results):
    """Print formatted error report"""
    stats = results['stats']
    
    print("\n" + "="*60)
    print("       OMISSION & COMMISSION ERROR REPORT")
    print("="*60)
    print(f"  Buffer Size: {stats['buffer_pixels']} pixels on each side")
    print("-"*60)
    print(f"  Ground Truth (HRDL) positive pixels: {stats['total_hrdl_pixels']:,}")
    print(f"  Prediction (L1PM) positive pixels:   {stats['total_l1pm_pixels']:,}")
    print("-"*60)
    print(f"  OMISSION ERROR (missed detections):")
    print(f"    Pixels: {stats['omission_pixels']:,}")
    print(f"    Rate:   {stats['omission_rate_percent']:.2f}%")
    print("-"*60)
    print(f"  COMMISSION ERROR (false detections):")
    print(f"    Pixels: {stats['commission_pixels']:,}")
    print(f"    Rate:   {stats['commission_rate_percent']:.2f}%")
    print("="*60)



    # ============= Full Image Inference =============
    print("\n--- Running Full Image Inference ---")
    
    # Load best model for inference
    inference_model = ImprovedCNN(dropout_rate=0.3)
    inference_model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    inference_model.to(device)
    inference_model.eval()
    
    with torch.no_grad():
        # Prepare input (normalize 16-bit to float)
        input_tensor = torch.from_numpy(msi_arr / 10000.0).float().unsqueeze(0).to(device)
        
        # Run prediction
        output = inference_model(input_tensor)
        binary_pred = (output > 0.5).cpu().numpy()[0, 0]  # L1PM
        
        # Compute standard metrics on full image
        target_tensor = torch.from_numpy(label_arr_resampled).float().unsqueeze(0).unsqueeze(0)
        metrics = compute_metrics(output.cpu(), target_tensor)
        
        print(f"\nFull Image Metrics (without buffer):")
        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Predicted positive pixels: {100 * np.mean(binary_pred):.2f}%")
        
        # ============= Omission & Commission Error Analysis =============
        print("\n--- Computing Omission & Commission Errors ---")
        
        # HRDL = Ground truth (label_arr_resampled)
        # L1PM = Prediction (binary_pred)
        hrdl = label_arr_resampled
        l1pm = binary_pred.astype(np.float32)
        
        # Compute errors with 2-pixel buffer
        error_results = compute_omission_commission_errors(hrdl, l1pm, buffer_pixels=2)
        
        # Print error report
        print_error_report(error_results)
        
        # Save all output files
        print("\n--- Saving Output Files ---")
        
        # Save L1PM (prediction)
        save_raster_tif('L1PM.tif', l1pm.astype(np.uint8), bbox, str(projection))
        print("  Saved: L1PM.tif (Prediction)")
        
        # Save HRDL buffered
        save_raster_tif('HRDL_buffered.tif', error_results['hrdl_buffered'].astype(np.uint8), 
                        bbox, str(projection))
        print("  Saved: HRDL_buffered.tif (Ground truth + 2px buffer)")
        
        # Save L1PM buffered
        save_raster_tif('L1PM_buffered.tif', error_results['l1pm_buffered'].astype(np.uint8), 
                        bbox, str(projection))
        print("  Saved: L1PM_buffered.tif (Prediction + 2px buffer)")
        
        # Save Omission Error map
        save_raster_tif('omission_error.tif', error_results['omission_error'].astype(np.uint8), 
                        bbox, str(projection))
        print("  Saved: omission_error.tif (HRDL - L1PM_buffered)")
        
        # Save Commission Error map
        save_raster_tif('commission_error.tif', error_results['commission_error'].astype(np.uint8), 
                        bbox, str(projection))
        print("  Saved: commission_error.tif (L1PM - HRDL_buffered)")
        
        print("\nAll files saved successfully!")


