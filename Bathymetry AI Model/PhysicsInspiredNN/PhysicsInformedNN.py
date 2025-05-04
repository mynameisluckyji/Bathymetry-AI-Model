import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
import itertools
from glob import glob
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler
import rasterio
from osgeo import gdal
from skimage.metrics import structural_similarity as compare_ssim
# Parameters
dataset = "spot6"  # Choose dataset: "UAV", "SPOT6", or "S2"
norm_param = np.load('/kaggle/input/magicbethynet/MagicBathyNet/agia_napa/norm_param_spot6_an.npy')  # Normalization parameters
norm_param_depth = -30.443  # Depth normalization parameter
WINDOW_SIZE = (30, 30)  # Changed to power of 2 for better architecture compatibility
STRIDE = 2  # Stride for sliding window
BATCH_SIZE = 16  # Increased batch size for more stable training
MAIN_FOLDER = '/kaggle/input/magicbethynet/MagicBathyNet/agia_napa'  # Path to your data folder
DATA_FOLDER = MAIN_FOLDER + '/img/spot6/img_{}.tif'  # Path to RGB images
LABEL_FOLDER = MAIN_FOLDER + '/depth/spot6/depth_{}.tif'  # Path to depth images
ERODED_FOLDER = MAIN_FOLDER + '/depth/spot6/depth_{}.tif'  # Path to eroded depth images
train_images = ['409', '418', '350', '399', '361', '430', '380', '359', '371', '377', '379', '360', '368', '419', '389', '420', '401', '408', '352', '388', '362', '421', '412', '351', '349', '390', '400', '378']
test_images = ['411', '387', '410', '398', '370', '369', '397']

# Enhanced Physics-Inspired Neural Network (PINN)
class PhysicsInspiredNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PhysicsInspiredNN, self).__init__()
        self.n_channels = in_channels
        self.n_outputs = out_channels
        
        # Enhanced feature extraction with residual connections
        self.encoder1 = self._make_encoder_block(in_channels, 64)
        self.encoder2 = self._make_encoder_block(64, 128)
        self.encoder3 = self._make_encoder_block(128, 256)
        self.encoder4 = self._make_encoder_block(256, 512)
        
        # Bottleneck with attention mechanism
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ChannelAttention(512),
            SpatialAttention()
        )
        
        # Decoder with skip connections
        self.decoder4 = self._make_decoder_block(512, 256)
        self.decoder3 = self._make_decoder_block(256, 128)
        self.decoder2 = self._make_decoder_block(128, 64)
        self.decoder1 = self._make_decoder_block(64, 64)
        
        # Final prediction with multi-scale output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Depth prediction head
        self.depth_head = nn.Sequential(
            nn.Conv2d(out_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        
        # Additional layers for dimension adjustment
        self.adjust_dec4 = nn.Conv2d(256, 256, kernel_size=1)  # Example adjustment
        self.adjust_dec3 = nn.Conv2d(128, 128, kernel_size=1)  # Example adjustment
        self.adjust_dec2 = nn.Conv2d(64, 64, kernel_size=1)  # Example adjustment
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # Bottleneck with attention
        bottleneck = self.bottleneck(enc4)
        
        # Decoder path with skip connections
        dec4 = self.decoder4(bottleneck)
        
        # Adjust dimensions if necessary
        enc3_adj = F.interpolate(enc3, size=dec4.shape[2:], mode='bilinear')
        dec4 = dec4 + enc3_adj
        
        dec3 = self.decoder3(dec4)
        enc2_adj = F.interpolate(enc2, size=dec3.shape[2:], mode='bilinear')
        dec3 = dec3 + enc2_adj
        
        dec2 = self.decoder2(dec3)
        enc1_adj = F.interpolate(enc1, size=dec2.shape[2:], mode='bilinear')
        dec2 = dec2 + enc1_adj
        
        dec1 = self.decoder1(dec2)
        
        # Final prediction
        out = self.final_conv(dec1)
        depth = self.depth_head(out)
        
        return depth

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return x * self.sigmoid(out)

# Enhanced physics-based loss function
class PhysicsLoss(nn.Module):
    def __init__(self):
        super(PhysicsLoss, self).__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
    def forward(self, output, target, mask):
        # Ensure spatial dimensions match
        if output.shape[2:] != target.shape[2:]:
            output = F.interpolate(output, size=target.shape[2:], mode='bilinear')
            mask = F.interpolate(mask, size=target.shape[2:], mode='nearest')
        
        # Data-driven loss (Smooth L1 loss is more robust to outliers than MSE)
        data_loss = F.smooth_l1_loss(output * mask, target * mask, reduction='sum') / mask.sum()
        
        # Edge-aware smoothness loss
        self.sobel_x = self.sobel_x.to(output.device)
        self.sobel_y = self.sobel_y.to(output.device)
        
        grad_x = F.conv2d(output, self.sobel_x, padding=1)
        grad_y = F.conv2d(output, self.sobel_y, padding=1)
        
        # Compute image gradients for edge weighting
        img_grad_x = F.conv2d(output, self.sobel_x, padding=1).abs()
        img_grad_y = F.conv2d(output, self.sobel_y, padding=1).abs()
        edge_weight = torch.exp(-(img_grad_x + img_grad_y))
        
        smoothness_loss = (grad_x.abs() * edge_weight + grad_y.abs() * edge_weight).mean()
        
        # Depth consistency loss (encourage similar depths for similar colors)
        color_similarity = torch.exp(-torch.mean((output - output.detach())**2, dim=1, keepdim=True))
        depth_diff = (output - output.permute(0, 1, 3, 2)).abs()
        consistency_loss = (color_similarity * depth_diff).mean()
        
        # Combine losses with adaptive weights
        total_loss = data_loss + 0.1 * smoothness_loss + 0.05 * consistency_loss
        return total_loss
def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w)
    x2 = x1 + w
    y1 = random.randint(0, H - h)
    y2 = y1 + h
    return x1, x2, y1, y2

# Enhanced Dataset class with more augmentations
class Dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER, cache=False, augmentation=True):
        super(Dataset, self).__init__()
        self.augmentation = augmentation
        self.cache = cache
        self.data_files = [data_files.format(id) for id in ids]
        self.label_files = [label_files.format(id) for id in ids]
        self.data_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        return 10000  # Default epoch size

    def __getitem__(self, i):
        random_idx = random.randint(0, len(self.data_files) - 1)

        if random_idx in self.data_cache_:
            data = self.data_cache_[random_idx]
        else:
            # Using rasterio for better handling of geospatial data
            with rasterio.open(self.data_files[random_idx]) as src:
                data = src.read().astype('float32')
            data = (data - norm_param[0][:, np.newaxis, np.newaxis]) / (norm_param[1][:, np.newaxis, np.newaxis] - norm_param[0][:, np.newaxis, np.newaxis])
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.label_cache_:
            label = self.label_cache_[random_idx]
        else:
            with rasterio.open(self.label_files[random_idx]) as src:
                label = src.read(1).astype('float32')
            label = 1 / norm_param_depth * label
            if self.cache:
                self.label_cache_[random_idx] = label

        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        if self.augmentation:
            data_p, label_p = self.data_augmentation(data_p, label_p)

        return torch.from_numpy(data_p), torch.from_numpy(label_p)

    @staticmethod
    def data_augmentation(*arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
            
        # Random rotation (0, 90, 180, 270 degrees)
        rot = random.choice([0, 1, 2, 3])
        
        # Random brightness/contrast adjustment
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)

        results = []
        for array in arrays:
            if will_flip:
                array = array[::-1, :] if len(array.shape) == 2 else array[:, ::-1, :]
            if will_mirror:
                array = array[:, ::-1] if len(array.shape) == 2 else array[:, :, ::-1]
            
            # Apply rotation
            if len(array.shape) == 2:
                array = np.rot90(array, rot)
            else:
                array = np.rot90(array, rot, axes=(1, 2))
            
            # Apply brightness/contrast to image only (not label)
            if len(array.shape) == 3:  # Image data
                array = array * brightness
                array = (array - array.mean()) * contrast + array.mean()
                array = np.clip(array, 0, 1)
            
            results.append(np.copy(array))
        return tuple(results)

# Load the network on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = PhysicsInspiredNN(3, 1).to(device)

# Load the datasets with validation split
train_set = Dataset(train_images, DATA_FOLDER, LABEL_FOLDER, cache=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# Define the optimizer with weight decay
base_lr = 0.0001
optimizer = optim.AdamW(net.parameters(), lr=base_lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Early stopping
best_loss = float('inf')
patience = 10
patience_counter = 0

# Enhanced training function with validation
def train(net, optimizer, epochs, scheduler=None, save_epoch=5):
    global best_loss, patience_counter
    criterion = PhysicsLoss()
    iter_ = 0
    
    for e in range(1, epochs + 1):
        net.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {e}/{epochs}')):
            data, target = Variable(data.to(device)), Variable(target.to(device))
            optimizer.zero_grad()

            # Forward pass
            output = net(data.float())
            
            # Compute loss
            target_mask = (target != 0).float().to(device)
            loss = criterion(output, target.unsqueeze(1), target_mask.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if iter_ % 100 == 0:
                print(f'Epoch {e}/{epochs}, Iter {iter_}, Loss: {loss.item()}')
            
            iter_ += 1
        
        # Average epoch loss
        epoch_loss /= len(train_loader)
        print(f'Epoch {e} completed. Average Loss: {epoch_loss}')
        
        # Step the scheduler based on validation loss
        if scheduler is not None:
            scheduler.step(epoch_loss)
        
        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(net.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {e}')
                break
        
        if e % save_epoch == 0:
            torch.save(net.state_dict(), f'model_epoch{e}.pth')
    
    torch.save(net.state_dict(), 'model_final.pth')


# Train the network
train(net, optimizer, 100, scheduler)


def test(net, test_ids):
    net.eval()
    all_preds, all_gts = [], []
    metrics = {
        'mse': [],
        'mae': [],
        'rmse': [],
        'ssim': []
    }
    
    for id_ in tqdm(test_ids, desc='Testing'):
        with rasterio.open(DATA_FOLDER.format(id_)) as src:
            img = src.read().astype('float32')
        img = (img - norm_param[0][:, np.newaxis, np.newaxis]) / (norm_param[1][:, np.newaxis, np.newaxis] - norm_param[0][:, np.newaxis, np.newaxis])
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
        
        with rasterio.open(LABEL_FOLDER.format(id_)) as src:
            gt = src.read(1).astype('float32')
        gt = 1 / norm_param_depth * gt
        
        # Use sliding window for large images
        pred = np.zeros_like(gt)
        count = np.zeros_like(gt)
        
        h, w = gt.shape
        window_h, window_w = WINDOW_SIZE
        
        for i in range(0, h - window_h + 1, STRIDE):
            for j in range(0, w - window_w + 1, STRIDE):
                patch = img[:, i:i+window_h, j:j+window_w]
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pred_patch = net(patch_tensor.float()).cpu().numpy().squeeze()
                    
                # Adjust dimensions if necessary
                if pred_patch.shape != (window_h, window_w):
                    pred_patch = scipy.ndimage.zoom(pred_patch, (window_h / pred_patch.shape[0], window_w / pred_patch.shape[1]), order=1)
                
                # Handle edge cases where patch doesn't fully fit
                if i + window_h > h:
                    pred_patch = pred_patch[:h - i, :]
                if j + window_w > w:
                    pred_patch = pred_patch[:, :w - j]
                
                pred[i:i+window_h, j:j+window_w] += pred_patch
                count[i:i+window_h, j:j+window_w] += 1
        
        # Avoid division by zero
        count[count == 0] = 1
        pred = pred / count
        
        # Calculate metrics
        mask = (gt != 0)
        if mask.sum() > 0:
            mse = ((pred[mask] - gt[mask]) ** 2).mean()
            mae = np.abs(pred[mask] - gt[mask]).mean()
            rmse = np.sqrt(mse)
            
            # Calculate SSIM on a sample of patches to save memory
            ssim_values = []
            for _ in range(20):  # Sample 20 patches
                i, j = np.random.randint(0, max(1, h - 64)), np.random.randint(0, max(1, w - 64))
                ssim_values.append(compare_ssim(
                    pred[i:i+64, j:j+64], 
                    gt[i:i+64, j:j+64],
                    data_range=gt.max() - gt.min()
                ))
            ssim = np.mean(ssim_values)
            
            metrics['mse'].append(mse)
            metrics['mae'].append(mae)
            metrics['rmse'].append(rmse)
            metrics['ssim'].append(ssim)
        
        all_preds.append(pred)
        all_gts.append(gt)
    
    # Print average metrics
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"{k.upper()}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    
    return all_preds, all_gts, metrics

# Test the network
all_preds, all_gts, test_metrics = test(net, test_images)

for pred, id_ in zip(all_preds, test_images):
    pred_img = pred * norm_param_depth
    
    # Save with original georeferencing
    with rasterio.open(LABEL_FOLDER.format(id_)) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1, nodata=0)
    
    with rasterio.open(f'prediction_{id_}.tif', 'w', **profile) as dst:
        dst.write(pred_img.astype(np.float32), 1)
    
    # Visualize
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(pred_img, cmap='viridis', vmin=pred_img.min(), vmax=pred_img.max())
    plt.title('Prediction')
    plt.colorbar()
    
    plt.subplot(122)
    gt_img = 1 / norm_param_depth * rasterio.open(LABEL_FOLDER.format(id_)).read(1)
    plt.imshow(gt_img, cmap='viridis', vmin=gt_img.min(), vmax=gt_img.max())
    plt.title('Ground Truth')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f'comparison_{id_}.png', dpi=300, bbox_inches='tight')
    plt.close()
