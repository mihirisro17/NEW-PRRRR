import sys
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


def load_raster_tif(filename):
    """Load GeoTIFF file and return numpy array"""
    with rasterio.open(filename) as src:
        array = src.read()
        bbox = src.bounds
        projection = src.crs
        transform = src.transform
    return array, bbox, projection, transform


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


def save_raster_tif(filename, array, bbox, projection='EPSG:4326'):
    """Save single or multi-band numpy array as GeoTIFF"""
    if array.ndim == 2:
        height, width = array.shape
        count = 1
        array = array.reshape(1, height, width)
    elif array.ndim == 3:
        count, height, width = array.shape
    else:
        raise ValueError(f"Array must be 2D or 3D, got shape {array.shape}")

    transform = from_bounds(*bbox, width, height)
    with rasterio.open(
        filename, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=array.dtype,
        crs=projection,
        transform=transform
    ) as dst:
        for i in range(count):
            dst.write(array[i], i + 1)


def extract_patches(image, mask, patch_size=128, stride=64):
    """
    Extract patches from image and mask with stride
    Returns lists of image patches and mask patches
    """
    _, H, W = image.shape
    h, w = patch_size, patch_size
    
    image_patches = []
    mask_patches = []
    
    for i in range(0, H - h + 1, stride):
        for j in range(0, W - w + 1, stride):
            img_patch = image[:, i:i+h, j:j+w]
            mask_patch = mask[i:i+h, j:j+w]
            
            # Only include patches with some positive samples (optional filter)
            if np.sum(mask_patch) > 0 or np.random.rand() < 0.3:  # Keep 30% of negative patches
                image_patches.append(img_patch)
                mask_patches.append(mask_patch)
    
    return image_patches, mask_patches


# ============= Early Stopping Class =============
class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
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
            self.best_weights = model.state_dict().copy()
            self.best_epoch = epoch
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()
            self.best_epoch = epoch
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            print(f"\nEarly stopping triggered! Best epoch: {self.best_epoch}, Best loss: {self.best_loss:.4f}")
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


# ============= Metrics =============
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


# ============= Improved CNN Model =============
class ImprovedCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ImprovedCNN, self).__init__()
        
        # 3x3 convolutions with batch normalization
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(64)

        # 1x1 convolutions
        # FIXED: Changed stride to 1 and added padding='same' to maintain 128x128 output size
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


# ============= Focal Loss for Class Imbalance =============
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ============= Custom Dataset with CutMix and Augmentation =============
class SegmentationDataset(Dataset):
    def __init__(self, images, masks, augment=False, cutmix_prob=0.5):
        """
        Args:
            images: List of numpy arrays (C, H, W)
            masks: List of numpy arrays (H, W)
            augment: Boolean to enable augmentation
            cutmix_prob: Probability of applying CutMix
        """
        self.images = images
        self.masks = masks
        self.augment = augment
        self.cutmix_prob = cutmix_prob

    def __len__(self):
        return len(self.images)

    def apply_cutmix(self, image, mask, current_idx):
        """
        Applies CutMix: Replaces a random patch of the current image/mask 
        with a patch from a random other image in the dataset.
        Handles Sentinel data by using pure numpy slicing (preserves bit depth).
        """
        # 1. Pick a random index that is not the current one
        rand_idx = np.random.randint(0, len(self.images))
        while rand_idx == current_idx:
            rand_idx = np.random.randint(0, len(self.images))
            
        other_image = self.images[rand_idx]
        other_mask = self.masks[rand_idx]
        
        # 2. Generate random bounding box
        C, H, W = image.shape
        
        # Lambda comes from Beta distribution, determines how big the box is
        # alpha=1.0 is uniform distribution
        lam = np.random.beta(1.0, 1.0)
        
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Center of the box
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # 3. Paste the 'other' image patch into the current image
        # Using .copy() is important to avoid modifying the original list data if cached
        image_aug = image.copy()
        mask_aug = mask.copy()
        
        image_aug[:, bby1:bby2, bbx1:bbx2] = other_image[:, bby1:bby2, bbx1:bbx2]
        mask_aug[bby1:bby2, bbx1:bbx2] = other_mask[bby1:bby2, bbx1:bbx2]
        
        return image_aug, mask_aug

    def __getitem__(self, idx):
        # We need to copy here to avoid modifying the global list during augs
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()

        # Data augmentation
        if self.augment:
            # 1. CutMix (Random replacement from another image)
            # Apply this BEFORE flips so the cut patch also gets flipped potentially,
            # or apply AFTER for a 'clean' cut. 
            # Applying it HERE (before normalization) ensures 16-bit/float data safety.
            if np.random.rand() < self.cutmix_prob:
                image, mask = self.apply_cutmix(image, mask, idx)

            # 2. Random horizontal flip (axis=2 for CHW image, axis=1 for HW mask)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            
            # 3. Random vertical flip (axis=1 for CHW image, axis=0 for HW mask)
            if np.random.rand() > 0.5:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()
            
            # 4. Random 90 degree rotations
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k=k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

        # Normalize AFTER augmentations to keep CutMix bit-depth safe
        image = image / 10000.0  
        
        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)

        return image_tensor, mask_tensor


# ============= Training Function with Metrics =============
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda', 
                use_focal_loss=False, early_stopping_patience=15):
    
    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("Using Focal Loss for class imbalance")
    else:
        criterion = nn.BCELoss()
        print("Using Binary Cross-Entropy Loss")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                      patience=5)
    
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
                
                # Compute metrics
                metrics = compute_metrics(outputs, masks)
                for key in val_metrics:
                    val_metrics[key] += metrics[key]

        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Track best model
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            torch.save(model.state_dict(), 'best_model.pth')

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_metrics['iou']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Early stopping check
        if early_stopping(val_loss, model, epoch):
            break

    return model


# ============= Main Execution =============
if __name__ == '__main__':

    # File paths (Adjust these to your local paths)
    msi_file = '/home/sac/Documents/urban_model/amd/MSI.tif'
    label_file = '/home/sac/Documents/urban_model/amd/T6S1P10.tif'
    
    # Check if files exist before running to prevent vague errors
    import os
    if not os.path.exists(msi_file):
        print(f"Error: File not found {msi_file}")
        sys.exit(1)
        
    print(f"Loading MSI data from {msi_file}...")
    msi_arr, bbox, projection, transform = load_raster_tif(msi_file)
    print(f"MSI array shape: {msi_arr.shape}")
    
    # Get MSI dimensions
    _, msi_height, msi_width = msi_arr.shape
    
    # Resample label to match MSI
    print(f"\nResampling label to match MSI dimensions...")
    label_arr_resampled = resample_to_match(
        label_file, 
        (msi_height, msi_width), 
        transform, 
        projection, 
        resampling_method=Resampling.nearest
    )
    
    if label_arr_resampled.ndim == 3 and label_arr_resampled.shape[0] == 1:
        label_arr_resampled = label_arr_resampled[0]
    
    label_arr_resampled = (label_arr_resampled > 0).astype(np.float32)
    print(f"Label shape: {label_arr_resampled.shape}")
    
    # Extract patches
    print("\n--- Extracting Patches ---")
    patch_size = 128
    stride = 64  # 50% overlap
    
    image_patches, mask_patches = extract_patches(msi_arr, label_arr_resampled, 
                                                   patch_size=patch_size, stride=stride)
    
    print(f"Total patches extracted: {len(image_patches)}")
    
    # Split into train and validation
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_patches, mask_patches, test_size=0.2, random_state=42
    )
    
    print(f"Training patches: {len(train_images)}")
    print(f"Validation patches: {len(val_images)}")
    
    # Create datasets with augmentation for training
    # Enabled cutmix_prob
    train_dataset = SegmentationDataset(train_images, train_masks, augment=True, cutmix_prob=0.5)
    val_dataset = SegmentationDataset(val_images, val_masks, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Initialize improved model
    model = ImprovedCNN(dropout_rate=0.3)
    
    # Check for CUDA
    device =  'cpu'
    print(f"\nUsing device: {device}")
    
    # Train model
    print("\n--- Training Model with Early Stopping and CutMix ---")
    trained_model = train_model(
        model, train_loader, val_loader, 
        epochs=500, 
        lr=0.001, 
        device=device,
        use_focal_loss=True,
        early_stopping_patience=30
    )
    
    # Save final model
    torch.save(trained_model.state_dict(), 'final_model_weights.pth')
    print('\nFinal model weights saved.')
    
    # Cleanup memory
    del train_images, val_images, train_loader, val_loader
    import gc
    gc.collect()

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
        binary_pred = (output > 0.5).cpu().numpy()[0, 0]
        
        # Compute metrics on full image
        target_tensor = torch.from_numpy(label_arr_resampled).float().unsqueeze(0).unsqueeze(0)
        metrics = compute_metrics(output.cpu(), target_tensor)
        
        print(f"\nFull Image Metrics:")
        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Predicted positive pixels: {100 * np.mean(binary_pred):.2f}%")
        
        # Save prediction
        save_raster_tif('prediction_best.tif', binary_pred.astype(np.uint8), bbox, str(projection))
        print("\nPrediction saved to prediction_best.tif")
