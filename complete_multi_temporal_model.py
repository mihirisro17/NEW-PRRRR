import sys
import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import reproject
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path


# ============= Data Loading & Alignment Functions =============

def load_raster_tif(filename):
    """Load GeoTIFF file and return numpy array with complete metadata"""
    with rasterio.open(filename) as src:
        array = src.read()
        bbox = src.bounds
        projection = src.crs
        transform = src.transform
        profile = src.profile
    return array, bbox, projection, transform, profile


def save_raster_tif(filename, array, transform, crs):
    """Save raster using EXPLICIT transform for perfect alignment"""
    if array.ndim == 2:
        height, width = array.shape
        count = 1
        array = array.reshape(1, height, width)
    elif array.ndim == 3:
        count, height, width = array.shape
    else:
        raise ValueError(f"Array must be 2D or 3D, got shape {array.shape}")

    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        compress='lzw'
    ) as dst:
        for i in range(count):
            dst.write(array[i], i + 1)
    print(f"  Saved: {filename}")


def resample_to_match(src_file, match_shape, match_transform, match_crs, resampling_method=Resampling.nearest):
    """Resample a raster to match the shape, transform, and CRS of another raster"""
    with rasterio.open(src_file) as src:
        dest_array = np.zeros((src.count, match_shape[0], match_shape[1]), dtype=src.dtypes[0])
        for i in range(src.count):
            reproject(
                source=rasterio.band(src, i + 1),
                destination=dest_array[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=match_transform,
                dst_crs=match_crs,
                resampling=resampling_method
            )
        return dest_array


def align_images_affine(image_ref, image_to_align):
    """Sub-pixel alignment using OpenCV ECC algorithm"""
    print("  -> Calculating Affine Transform for precise alignment...")

    # Ensure 2D images for alignment
    if image_ref.ndim == 3:
        image_ref = np.mean(image_ref, axis=0)
    if image_to_align.ndim == 3:
        image_to_align = image_to_align[0]

    # Convert to float32 and normalize
    im1 = image_ref.astype(np.float32)
    im2 = image_to_align.astype(np.float32)

    im1 = (im1 - np.min(im1)) / (np.max(im1) - np.min(im1) + 1e-7)
    im2 = (im2 - np.min(im2)) / (np.max(im2) - np.min(im2) + 1e-7)

    # Convert to uint8 for better ECC performance
    im1_uint8 = (im1 * 255).astype(np.uint8)
    im2_uint8 = (im2 * 255).astype(np.uint8)

    # Initialize warp matrix
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)

    try:
        cc, warp_matrix = cv2.findTransformECC(im1_uint8, im2_uint8, warp_matrix, warp_mode, criteria)
        print(f"  -> Alignment converged (correlation: {cc:.4f})")

        sz = image_ref.shape
        aligned_image = cv2.warpAffine(
            image_to_align, 
            warp_matrix, 
            (sz[1], sz[0]),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP
        )
        return aligned_image

    except cv2.error as e:
        print(f"  -> Warning: Alignment failed ({e}). Using unaligned image.")
        return image_to_align


def load_and_process_region(msi_file, label_file, region_name, require_label=True, save_aligned=True, output_dir='align'):
    """
    Load and process a single region's MSI and label data.

    Args:
        msi_file: Path to MSI file
        label_file: Path to label file (can be None if require_label=False)
        region_name: Name of the region
        require_label: If False, label is optional (for inference-only)
        save_aligned: If True, save aligned label to output_dir
        output_dir: Directory to save aligned labels

    Returns:
        Dictionary with msi, label (or None), transform, crs, bbox, region_name
    """
    print(f"\n{'='*60}")
    print(f"Processing Region: {region_name.upper()}")
    print(f"{'='*60}")

    # Check MSI file existence
    if not os.path.exists(msi_file):
        print(f"  ERROR: MSI file not found: {msi_file}")
        return None

    # Check label file existence
    has_label = False
    if label_file is not None and os.path.exists(label_file):
        has_label = True
    elif require_label:
        print(f"  ERROR: Label file not found: {label_file}")
        return None
    else:
        print(f"  INFO: No label file provided (inference-only mode)")

    # Load MSI
    print(f"  Loading MSI: {Path(msi_file).name}")
    msi_arr, bbox, projection, transform, profile = load_raster_tif(msi_file)
    print(f"    Shape: {msi_arr.shape}, Range: [{msi_arr.min()}, {msi_arr.max()}]")

    _, msi_height, msi_width = msi_arr.shape

    # Process label if available
    label_final = None
    if has_label:
        # Resample label
        print(f"  Loading Label: {Path(label_file).name}")
        label_arr_resampled = resample_to_match(
            label_file, 
            (msi_height, msi_width), 
            transform, 
            projection, 
            resampling_method=Resampling.nearest
        )

        if label_arr_resampled.ndim == 3 and label_arr_resampled.shape[0] == 1:
            label_arr_resampled = label_arr_resampled[0]

        # Affine alignment
        print(f"  Performing affine alignment...")
        msi_ref = np.mean(msi_arr, axis=0)
        label_aligned = align_images_affine(msi_ref, label_arr_resampled)

        # Binarize label
        label_final = (label_aligned > 0).astype(np.float32)

        # Statistics
        class_counts = np.bincount(label_final.astype(int).flatten())
        total_pixels = class_counts.sum()
        print(f"  Class distribution:")
        print(f"    Background: {class_counts[0]:,} ({100*class_counts[0]/total_pixels:.2f}%)")
        print(f"    Target:     {class_counts[1]:,} ({100*class_counts[1]/total_pixels:.2f}%)")

        # Save aligned label if requested
        if save_aligned:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Save aligned label
            aligned_filename = os.path.join(output_dir, f'{region_name}_label_aligned.tif')
            save_raster_tif(aligned_filename, label_final.astype(np.uint8), transform, projection)
            print(f"  Aligned label saved to: {aligned_filename}")

    return {
        'msi': msi_arr,
        'label': label_final,  # Will be None if no label provided
        'transform': transform,
        'crs': projection,
        'bbox': bbox,
        'region_name': region_name,
        'has_label': has_label
    }


def extract_patches(image, mask, patch_size=128, stride=64):
    """Extract patches from image and mask with stride"""
    _, H, W = image.shape
    h, w = patch_size, patch_size

    image_patches = []
    mask_patches = []

    for i in range(0, H - h + 1, stride):
        for j in range(0, W - w + 1, stride):
            img_patch = image[:, i:i+h, j:j+w]
            mask_patch = mask[i:i+h, j:j+w]

            # Keep all positive patches + 30% of negative patches
            if np.sum(mask_patch) > 0 or np.random.rand() < 0.3:
                image_patches.append(img_patch)
                mask_patches.append(mask_patch)

    return image_patches, mask_patches


# ============= Early Stopping & Metrics =============

class EarlyStopping:
    """Early stopping with model checkpoint restoration"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.best_weights = None
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"\nEarly stopping triggered! Best epoch: {self.best_epoch + 1}, Best loss: {self.best_loss:.4f}")
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union (IoU)"""
    pred = (pred > threshold).float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.item()


def compute_metrics(pred, target, threshold=0.5):
    """Compute accuracy, precision, recall, F1, and IoU"""
    pred_binary = (pred > threshold).float()
    target = target.float()

    tp = ((pred_binary == 1) & (target == 1)).sum().float()
    tn = ((pred_binary == 0) & (target == 0)).sum().float()
    fp = ((pred_binary == 1) & (target == 0)).sum().float()
    fn = ((pred_binary == 0) & (target == 1)).sum().float()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)

    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'iou': iou.item()
    }


# ============= Improved CNN Model (FIXED) =============

class ImprovedCNN(nn.Module):
    """Simplified CNN with fixed spatial dimensions"""
    def __init__(self, dropout_rate=0.3):
        super(ImprovedCNN, self).__init__()

        # 3x3 convolutions with batch normalization
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(64)

        # FIXED: stride=1 and padding='same' to maintain spatial dimensions
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same')
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 16, kernel_size=1, stride=1)
        self.bn5 = nn.BatchNorm2d(16)

        self.conv6 = nn.Conv2d(16, 1, kernel_size=1, stride=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.sigmoid(self.conv6(x))
        return x


# ============= Focal Loss =============

class FocalLoss(nn.Module):
    """Focal Loss to handle severe class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ============= Dataset with CutMix =============

class SegmentationDataset(Dataset):
    """Dataset with advanced augmentation including CutMix"""
    def __init__(self, images, masks, augment=False, cutmix_prob=0.5, normalization_factor=10000.0):
        self.images = images
        self.masks = masks
        self.augment = augment
        self.cutmix_prob = cutmix_prob
        self.normalization_factor = normalization_factor

    def __len__(self):
        return len(self.images)

    def apply_cutmix(self, image, mask, current_idx):
        """CutMix augmentation"""
        rand_idx = np.random.randint(0, len(self.images))
        while rand_idx == current_idx and len(self.images) > 1:
            rand_idx = np.random.randint(0, len(self.images))

        other_image = self.images[rand_idx]
        other_mask = self.masks[rand_idx]

        C, H, W = image.shape
        lam = np.random.beta(1.0, 1.0)
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        image_aug = image.copy()
        mask_aug = mask.copy()

        image_aug[:, bby1:bby2, bbx1:bbx2] = other_image[:, bby1:bby2, bbx1:bbx2]
        mask_aug[bby1:bby2, bbx1:bbx2] = other_mask[bby1:bby2, bbx1:bbx2]

        return image_aug, mask_aug

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()

        if self.augment:
            if np.random.rand() < self.cutmix_prob:
                image, mask = self.apply_cutmix(image, mask, idx)

            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()

            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()

            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k=k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

        # Normalize
        image = image / self.normalization_factor
        image = np.clip(image, 0, 1)

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return image_tensor, mask_tensor


# ============= Training Function =============

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda',
                use_focal_loss=True, early_stopping_patience=15, model_save_path='best_model.pth'):
    """Train segmentation model with early stopping"""

    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using Focal Loss for class imbalance")
    else:
        criterion = nn.BCELoss()
        print("Using Binary Cross-Entropy Loss")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=0.001)

    model.to(device)
    best_val_iou = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_iou = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += compute_iou(outputs, masks)

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0}

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                metrics = compute_metrics(outputs, masks)
                for key in val_metrics:
                    val_metrics[key] += metrics[key]

        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Track best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save(model.state_dict(), model_save_path)

        print(f"Epoch [{epoch+1}/{epochs}] LR: {current_lr:.6f}")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}")

        if early_stopping(val_loss, model, epoch):
            break

    print(f"\nTraining completed. Best validation IoU: {best_val_iou:.4f}")
    return model


# ============= Inference Function (UPDATED) =============

def run_inference(model, msi_arr, label_arr, transform, crs, output_prefix, device='cuda', 
                  normalization_factor=10000.0, save_outputs=True):
    """
    Run inference on full image and save outputs.

    Args:
        label_arr: Can be None if no ground truth available
        save_outputs: If True, save prediction and probability files

    Returns:
        Dictionary containing metrics (if label available) and probability map
    """

    print(f"\n{'='*60}")
    print(f"Running Inference: {output_prefix}")
    print(f"{'='*60}")

    model.eval()

    with torch.no_grad():
        # Normalize input
        input_tensor = msi_arr / normalization_factor
        input_tensor = np.clip(input_tensor, 0, 1)
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(device)

        print(f"  Input shape: {input_tensor.shape}")

        # Run inference
        output = model(input_tensor)

        # Get predictions
        prob_map = output.cpu().numpy()[0, 0]
        binary_pred = (prob_map > 0.5).astype(np.uint8)

        # Compute metrics only if label is available
        metrics = None
        if label_arr is not None:
            target_tensor = torch.from_numpy(label_arr).float().unsqueeze(0).unsqueeze(0)
            metrics = compute_metrics(output.cpu(), target_tensor)

            print(f"\n  Metrics:")
            print(f"    IoU:       {metrics['iou']:.4f}")
            print(f"    F1 Score:  {metrics['f1']:.4f}")
            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")

            total_pixels = label_arr.size
            true_positive = np.sum(label_arr)
            print(f"\n    Ground truth positive: {int(true_positive):,} ({100*true_positive/total_pixels:.2f}%)")
        else:
            print(f"\n  No ground truth available - skipping metrics calculation")

        # Prediction statistics (always available)
        total_pixels = binary_pred.size
        pred_positive = np.sum(binary_pred)
        print(f"    Predicted positive:    {int(pred_positive):,} ({100*pred_positive/total_pixels:.2f}%)")

        # Save outputs if requested
        if save_outputs:
            save_raster_tif(f'{output_prefix}_prediction.tif', binary_pred, transform, crs)
            save_raster_tif(f'{output_prefix}_probability.tif', prob_map.astype(np.float32), transform, crs)

        return {
            'metrics': metrics,
            'probability': prob_map,
            'prediction': binary_pred,
            'transform': transform,
            'crs': crs
        }


# ============= Difference Computation Function =============

def compute_probability_difference(later_result, earlier_result, diff_name, description):
    """
    Compute difference between two probability maps and save result.

    Args:
        later_result: Inference result dictionary for later date
        earlier_result: Inference result dictionary for earlier date
        diff_name: Name for the difference file
        description: Description of the difference computation
    """
    print(f"\n{'='*60}")
    print(f"Computing Probability Difference: {description}")
    print(f"{'='*60}")

    prob_later = later_result['probability']
    prob_earlier = earlier_result['probability']

    # Check if shapes match
    if prob_later.shape != prob_earlier.shape:
        print(f"  ERROR: Shape mismatch!")
        print(f"    Later:   {prob_later.shape}")
        print(f"    Earlier: {prob_earlier.shape}")
        return None

    # Compute difference
    diff_map = prob_later - prob_earlier

    # Statistics
    print(f"  Difference statistics:")
    print(f"    Min:    {diff_map.min():.4f}")
    print(f"    Max:    {diff_map.max():.4f}")
    print(f"    Mean:   {diff_map.mean():.4f}")
    print(f"    Std:    {diff_map.std():.4f}")

    # Count pixels with significant change
    increase = np.sum(diff_map > 0.1)
    decrease = np.sum(diff_map < -0.1)
    no_change = np.sum(np.abs(diff_map) <= 0.1)
    total = diff_map.size

    print(f"\n  Change analysis (threshold=0.1):")
    print(f"    Increase:   {increase:,} ({100*increase/total:.2f}%)")
    print(f"    Decrease:   {decrease:,} ({100*decrease/total:.2f}%)")
    print(f"    No change:  {no_change:,} ({100*no_change/total:.2f}%)")

    # Save difference map
    output_file = f'{diff_name}.tif'
    save_raster_tif(
        output_file, 
        diff_map.astype(np.float32), 
        later_result['transform'], 
        later_result['crs']
    )

    return diff_map


# ============= Main Execution =============

if __name__ == '__main__':

    # ===== Configuration =====
    DATA_DIR = '/home/sac/Documents/urban_model/amd/multi_region/data'

    # Training regions
    TRAIN_REGIONS = {
        'ahmedabad': {
            'msi': 'MSI_ahmedabad.tif',
            'label': 'T6S1P10_ahmedabad.tif'
        },
        'gandhinagar': {
            'msi': 'MSI_gandhinagar.tif',
            'label': 'T6S1P10_gandhinagar.tif'
        },
        'bhuj': {
            'msi': 'MSI_bhuj.tif',
            'label': 'T6S1P10_bhuj.tif'
        }
    }

    # Validation region (spatial)
    VAL_REGION = {
        'vadodara': {
            'msi': 'MSI_vadodara.tif',
            'label': 'T6S1P10_vadodara.tif'
        }
    }

    # Temporal test files (multiple dates)
    TEMPORAL_TEST = {
        'ahmedabad_20240416': {
            'msi': 'MSI_ahmedabad_20240416.tif',
            'label': 'T6S1P10_ahmedabad_20240416.tif'  # Has ground truth
        },
        'ahmedabad_20240501': {
            'msi': 'MSI_ahmedabad_20240501.tif',
            'label': None
        },
        'ahmedabad_20250416': {
            'msi': 'MSI_ahmedabad_20250416.tif',
            'label': None
        },
        'ahmedabad_20250521': {
            'msi': 'MSI_ahmedabad_20250521.tif',
            'label': None
        },
        'ahmedabad_2025': {
            'msi': 'MSI_2025_ahmedabad.tif',
            'label': None
        }
    }

    # Difference pairs to compute
    DIFFERENCE_PAIRS = [
        {
            'name': 'diff_202505_202405',
            'later': 'ahmedabad_20250521',
            'earlier': 'ahmedabad_20240501',
            'description': 'May 2025 - May 2024'
        },
        {
            'name': 'diff_202504_202404',
            'later': 'ahmedabad_20250416',
            'earlier': 'ahmedabad_20240416',
            'description': 'April 2025 - April 2024'
        }
    ]

    # Hyperparameters
    PATCH_SIZE = 128
    STRIDE = 64
    BATCH_SIZE = 8
    EPOCHS = 500
    LEARNING_RATE = 0.001
    EARLY_STOP_PATIENCE = 50
    DROPOUT_RATE = 0.3
    CUTMIX_PROB = 0.5
    NORMALIZATION_FACTOR = 10000.0
    ALIGN_DIR = '/home/sac/Documents/urban_model/amd/multi_region/align'

    device = 'cuda'
    print(f"\n{'='*60}")
    print(f"MULTI-REGION TRAINING PIPELINE")
    print(f"{'='*60}")
    print(f"Device: {device}")

    # ===== Step 1: Load and Process All Training Regions =====
    print(f"\n{'='*60}")
    print("STEP 1: LOADING TRAINING REGIONS")
    print(f"{'='*60}")

    train_data_list = []
    for region_name, files in TRAIN_REGIONS.items():
        msi_path = os.path.join(DATA_DIR, files['msi'])
        label_path = os.path.join(DATA_DIR, files['label'])

        region_data = load_and_process_region(
            msi_path, label_path, region_name, 
            require_label=True, 
            save_aligned=True,
            output_dir=ALIGN_DIR
        )
        if region_data is not None:
            train_data_list.append(region_data)

    if len(train_data_list) == 0:
        print("ERROR: No training data loaded!")
        sys.exit(1)

    print(f"\nLoaded {len(train_data_list)} training regions successfully!")

    # ===== Step 2: Extract Patches from All Training Regions =====
    print(f"\n{'='*60}")
    print("STEP 2: EXTRACTING PATCHES FROM TRAINING REGIONS")
    print(f"{'='*60}")

    all_train_patches = []
    all_train_masks = []

    for region_data in train_data_list:
        print(f"\nExtracting patches from: {region_data['region_name']}")
        img_patches, mask_patches = extract_patches(
            region_data['msi'], 
            region_data['label'], 
            patch_size=PATCH_SIZE, 
            stride=STRIDE
        )
        print(f"  Extracted {len(img_patches)} patches")
        all_train_patches.extend(img_patches)
        all_train_masks.extend(mask_patches)

    print(f"\nTotal training patches: {len(all_train_patches)}")

    # ===== Step 3: Load Validation Region (Vadodara) =====
    print(f"\n{'='*60}")
    print("STEP 3: LOADING VALIDATION REGION (VADODARA)")
    print(f"{'='*60}")

    val_data = None
    for region_name, files in VAL_REGION.items():
        msi_path = os.path.join(DATA_DIR, files['msi'])
        label_path = os.path.join(DATA_DIR, files['label'])
        val_data = load_and_process_region(
            msi_path, label_path, region_name, 
            require_label=True,
            save_aligned=True,
            output_dir=ALIGN_DIR
        )
        break

    if val_data is None:
        print("ERROR: Validation data not loaded!")
        sys.exit(1)

    # Extract validation patches
    print(f"\nExtracting validation patches...")
    val_patches, val_masks = extract_patches(
        val_data['msi'], 
        val_data['label'], 
        patch_size=PATCH_SIZE, 
        stride=STRIDE
    )
    print(f"  Validation patches: {len(val_patches)}")

    # ===== Step 4: Create Datasets and Loaders =====
    print(f"\n{'='*60}")
    print("STEP 4: CREATING DATASETS")
    print(f"{'='*60}")

    train_dataset = SegmentationDataset(
        all_train_patches, all_train_masks, 
        augment=True, 
        cutmix_prob=CUTMIX_PROB,
        normalization_factor=NORMALIZATION_FACTOR
    )
    val_dataset = SegmentationDataset(
        val_patches, val_masks, 
        augment=False,
        normalization_factor=NORMALIZATION_FACTOR
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    # ===== Step 5: Initialize Model =====
    print(f"\n{'='*60}")
    print("STEP 5: MODEL INITIALIZATION")
    print(f"{'='*60}")

    model = ImprovedCNN(dropout_rate=DROPOUT_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ===== Step 6: Train Model =====
    print(f"\n{'='*60}")
    print("STEP 6: TRAINING MODEL")
    print(f"{'='*60}")

    trained_model = train_model(
        model, train_loader, val_loader, 
        epochs=EPOCHS, 
        lr=LEARNING_RATE, 
        device=device,
        use_focal_loss=True,
        early_stopping_patience=EARLY_STOP_PATIENCE,
        model_save_path='best_model_multi_region.pth'
    )

    torch.save(trained_model.state_dict(), 'final_model_multi_region.pth')
    print('\n  Models saved:')
    print('    - best_model_multi_region.pth (best validation)')
    print('    - final_model_multi_region.pth (final state)')

    # Cleanup memory
    del all_train_patches, all_train_masks, val_patches, val_masks
    del train_loader, val_loader, train_dataset, val_dataset
    import gc
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # ===== Step 7: Inference on Validation Region (Vadodara) =====
    print(f"\n{'='*60}")
    print("STEP 7: SPATIAL VALIDATION - VADODARA")
    print(f"{'='*60}")

    inference_model = ImprovedCNN(dropout_rate=DROPOUT_RATE)
    inference_model.load_state_dict(torch.load('best_model_multi_region.pth', weights_only=True))
    inference_model.to(device)

    vadodara_result = run_inference(
        inference_model,
        val_data['msi'],
        val_data['label'],
        val_data['transform'],
        val_data['crs'],
        'vadodara',
        device=device,
        normalization_factor=NORMALIZATION_FACTOR,
        save_outputs=True
    )
    vadodara_metrics = vadodara_result['metrics']

    # ===== Step 8: Temporal Test (Multiple Dates) =====
    print(f"\n{'='*60}")
    print("STEP 8: TEMPORAL ANALYSIS - MULTIPLE DATES")
    print(f"{'='*60}")

    # Dictionary to store inference results
    temporal_results = {}

    # Process all temporal test files
    for region_name, files in TEMPORAL_TEST.items():
        msi_path = os.path.join(DATA_DIR, files['msi'])
        label_path = os.path.join(DATA_DIR, files['label']) if files['label'] else None

        if not os.path.exists(msi_path):
            print(f"\n  WARNING: File not found: {msi_path}")
            continue

        # Check if label exists (if specified)
        has_label = False
        if label_path and os.path.exists(label_path):
            has_label = True
        elif label_path and not os.path.exists(label_path):
            print(f"  WARNING: Label file not found: {label_path}, proceeding without it")
            label_path = None

        # Load data
        temporal_data = load_and_process_region(
            msi_path, 
            label_path, 
            region_name, 
            require_label=False,
            save_aligned=has_label,
            output_dir=ALIGN_DIR
        )

        if temporal_data is not None:
            # Run inference and store results
            result = run_inference(
                inference_model,
                temporal_data['msi'],
                temporal_data['label'],
                temporal_data['transform'],
                temporal_data['crs'],
                region_name,
                device=device,
                normalization_factor=NORMALIZATION_FACTOR,
                save_outputs=True
            )
            temporal_results[region_name] = result

    print(f"\n  Completed inference on {len(temporal_results)} temporal files")

    # ===== Step 9: Compute Probability Differences =====
    print(f"\n{'='*60}")
    print("STEP 9: COMPUTING PROBABILITY DIFFERENCES")
    print(f"{'='*60}")

    difference_maps = {}

    for diff_pair in DIFFERENCE_PAIRS:
        later_key = diff_pair['later']
        earlier_key = diff_pair['earlier']
        diff_name = diff_pair['name']
        description = diff_pair['description']

        # Check if both results are available
        if later_key not in temporal_results:
            print(f"\n  WARNING: Missing data for {later_key}, skipping {diff_name}")
            continue
        if earlier_key not in temporal_results:
            print(f"\n  WARNING: Missing data for {earlier_key}, skipping {diff_name}")
            continue

        # Compute difference
        diff_map = compute_probability_difference(
            temporal_results[later_key],
            temporal_results[earlier_key],
            diff_name,
            description
        )

        if diff_map is not None:
            difference_maps[diff_name] = diff_map

    print(f"\n  Computed {len(difference_maps)} difference maps")

    # ===== Step 10: Inference on All Training Regions (Sanity Check) =====
    print(f"\n{'='*60}")
    print("STEP 10: INFERENCE ON TRAINING REGIONS (SANITY CHECK)")
    print(f"{'='*60}")

    training_metrics = {}
    for region_data in train_data_list:
        result = run_inference(
            inference_model,
            region_data['msi'],
            region_data['label'],
            region_data['transform'],
            region_data['crs'],
            region_data['region_name'],
            device=device,
            normalization_factor=NORMALIZATION_FACTOR,
            save_outputs=True
        )
        training_metrics[region_data['region_name']] = result['metrics']

    # ===== Final Summary =====
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")

    print("\n Training Regions Performance:")
    for region_name, metrics in training_metrics.items():
        print(f"  {region_name.capitalize():20s} - IoU: {metrics['iou']:.4f}, F1: {metrics['f1']:.4f}")

    print(f"\n Spatial Validation (Vadodara):")
    print(f"  {'Vadodara':20s} - IoU: {vadodara_metrics['iou']:.4f}, F1: {vadodara_metrics['f1']:.4f}")

    print(f"\n Temporal Analysis Results:")
    for region_name, result in temporal_results.items():
        if result['metrics'] is not None:
            print(f"  {region_name:20s} - IoU: {result['metrics']['iou']:.4f}, F1: {result['metrics']['f1']:.4f}")
        else:
            print(f"  {region_name:20s} - Prediction completed (no ground truth)")

    print(f"\n Difference Maps Computed:")
    for diff_name, description in [(p['name'], p['description']) for p in DIFFERENCE_PAIRS]:
        if diff_name in difference_maps:
            print(f"  {diff_name:25s} - {description}")

    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*60}")

    print("\nGenerated files:")
    print("  Models:")
    print("    - best_model_multi_region.pth")
    print("    - final_model_multi_region.pth")

    print("\n  Training Region Predictions:")
    for region_data in train_data_list:
        print(f"    - {region_data['region_name']}_prediction.tif")
        print(f"    - {region_data['region_name']}_probability.tif")

    print(f"\n  Validation Predictions:")
    print(f"    - vadodara_prediction.tif")
    print(f"    - vadodara_probability.tif")

    print(f"\n  Temporal Predictions:")
    for region_name in temporal_results.keys():
        print(f"    - {region_name}_prediction.tif")
        print(f"    - {region_name}_probability.tif")

    print(f"\n  Difference Maps:")
    for diff_name in difference_maps.keys():
        print(f"    - {diff_name}.tif")

    print(f"\n  Aligned Labels (saved in '{ALIGN_DIR}/'):")
    print(f"    - Training and validation regions with ground truth")
