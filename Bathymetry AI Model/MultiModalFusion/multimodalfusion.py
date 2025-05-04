import os
import gc
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
import rasterio
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
from torchvision import transforms
from tqdm import tqdm

# Configuration with analyzed parameters
class Config:
    def __init__(self):
        self.data_root = '/kaggle/input/magicbethynet/MagicBathyNet'
        self.site = 'agia_napa'
        self.modalities = ['aerial', 's2', 'spot6']
        self.train_ids = ['409', '418', '350', '399', '361', '430', '380', '359', '371', '377', '379', '360', '368', '419', '389', '420', '401', '408', '352', '388', '362', '421', '412', '351', '349', '390', '400', '378']
        self.test_ids = ['411', '387', '410', '398', '370', '369', '397']
        self.batch_size = 2
        self.patch_size = 16
        self.emb_size = 128
        self.num_heads = 2
        self.depth = 2
        self.forward_expansion = 2
        self.dropout = 0.1
        self.lr = 0.0001
        self.epochs = 50
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.grad_accum_steps = 8
        self.use_checkpointing = True
        
        # Load normalization parameters
        self.norm_params = {
            'aerial': np.load(os.path.join(self.data_root, self.site, 'norm_param_aerial.npy')),
            's2': np.load(os.path.join(self.data_root, self.site, 'norm_param_s2_an.npy')),
            'spot6': np.load(os.path.join(self.data_root, self.site, 'norm_param_spot6_an.npy'))
        }
        
        # Calculate proper depth normalization
        self.norm_param_depth = self.calculate_depth_stats()
    
    def calculate_depth_stats(self):
        """Calculate proper depth normalization parameters"""
        depths = []
        for sample_id in self.train_ids + self.test_ids:
            path = os.path.join(self.data_root, self.site, 'depth', self.modalities[0], f'depth_{sample_id}.tif')
            with rasterio.open(path) as src:
                depths.append(src.read(1, out_dtype=np.float32))
        depths = np.concatenate([d.flatten() for d in depths])
        print(f"Depth stats - Min: {depths.min():.2f}, Max: {depths.max():.2f}, Mean: {depths.mean():.2f}")
        return depths.max() - depths.min()  # Or use 2*std for better normalization

config = Config()

class MultimodalBathymetryDataset(Dataset):
    def __init__(self, ids, config, mode='train'):
        self.config = config
        self.ids = ids
        self.mode = mode
        self.cache = {}
        self.resize = transforms.Resize((256, 256))
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        
        if sample_id in self.cache:
            return self.cache[sample_id]
        
        # Load modalities
        modalities_data = []
        for modality in self.config.modalities:
            img_path = os.path.join(self.config.data_root, self.config.site, 'img', modality, f'img_{sample_id}.tif')
            with rasterio.open(img_path) as src:
                img = src.read(out_dtype=np.float32).transpose(1, 2, 0)
            
            # Normalize and transform
            img = (img - self.config.norm_params[modality][0]) / (self.config.norm_params[modality][1] - self.config.norm_params[modality][0])
            img = np.clip(img, 0, 1)
            img_tensor = self.input_transform(img).float()
            img_tensor = self.resize(img_tensor)
            modalities_data.append(img_tensor)
        
        # Load and process depth
        depth_path = os.path.join(self.config.data_root, self.config.site, 'depth', self.config.modalities[0], f'depth_{sample_id}.tif')
        with rasterio.open(depth_path) as src:
            depth = src.read(1, out_dtype=np.float32)
        
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        depth_tensor = self.resize(depth_tensor)
        depth_tensor = depth_tensor / self.config.norm_param_depth  # Normalize to [0,1]
        
        # Augmentation
        if self.mode == 'train' and random.random() > 0.5:
            modalities_data = [torch.flip(m, [2]) for m in modalities_data]
            depth_tensor = torch.flip(depth_tensor, [2])
        
        result = (modalities_data, depth_tensor)
        self.cache[sample_id] = result
        return result

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * config.forward_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * config.forward_expansion, emb_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ff(x))
        return x

class MultimodalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64, 64))
        )
        
        # Transformer components
        self.patch_embed = nn.Linear(64, config.emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, 64*64 + 1, config.emb_size))
        self.transformer = nn.Sequential(*[
            TransformerBlock(config.emb_size, config.num_heads, config.dropout)
            for _ in range(config.depth)
        ])
        
        # Decoder with residual connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.emb_size, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1))
    
    def forward(self, x_list):
        # Encode all modalities
        features = []
        for x in x_list:
            f = checkpoint(self.shared_encoder, x) if self.training and config.use_checkpointing else self.shared_encoder(x)
            b, c, h, w = f.shape
            f = f.permute(0, 2, 3, 1).reshape(b, h*w, c)
            features.append(f)
        
        # Average features
        x = torch.stack(features).mean(dim=0)
        x = self.patch_embed(x)
        
        # Add positional embeddings
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embed
        
        # Transformer
        x = checkpoint(self.transformer, x) if self.training and config.use_checkpointing else self.transformer(x)
        
        # Prepare for decoding
        cls_token = x[:, 0]
        spatial = x[:, 1:].reshape(b, 64, 64, -1).permute(0, 3, 1, 2)
        cls_token = cls_token.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 64, 64)
        x = spatial + cls_token
        
        # Decode to 256x256
        return self.decoder(x)

def train():
    torch.cuda.empty_cache()
    gc.collect()
    
    train_dataset = MultimodalBathymetryDataset(config.train_ids, config, mode='train')
    test_dataset = MultimodalBathymetryDataset(config.test_ids, config, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                            pin_memory=True, num_workers=4, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    model = MultimodalTransformer(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    scaler = amp.GradScaler()
    
    best_loss = float('inf')
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        for batch_idx, (modalities, depth) in enumerate(pbar):
            modalities = [m.to(config.device, non_blocking=True) for m in modalities]
            depth = depth.to(config.device, non_blocking=True)
            
            with amp.autocast():
                output = model(modalities)
                loss = F.mse_loss(output, depth) / config.grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * config.grad_accum_steps
            pbar.set_postfix({'Loss': f"{epoch_loss/(batch_idx+1):.4f}"})
        
        # Validate
        test_loss = evaluate(model, test_loader, config, full_eval=False)
        scheduler.step(test_loss)
        
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with test loss: {best_loss:.4f}")

def evaluate(model, test_loader, config, full_eval=True):
    model.eval()
    total_loss = 0
    preds, truths = [], []
    
    with torch.no_grad():
        for modalities, depth in tqdm(test_loader, desc='Evaluating'):
            modalities = [m.to(config.device) for m in modalities]
            depth = depth.to(config.device)
            
            output = model(modalities)
            loss = F.mse_loss(output, depth)
            total_loss += loss.item()
            
            if full_eval:
                # Denormalize
                pred_depth = output * config.norm_param_depth
                true_depth = depth * config.norm_param_depth
                preds.append(pred_depth.cpu().numpy())
                truths.append(true_depth.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    
    if full_eval:
        preds = np.concatenate(preds)
        truths = np.concatenate(truths)
        errors = truths - preds
        
        metrics = {
            'MAE': np.mean(np.abs(errors)),
            'MSE': np.mean(errors**2),
            'RMSE': np.sqrt(np.mean(errors**2)),
            'STD': np.std(errors),
            'MedianAE': np.median(np.abs(errors)),
            'MaxError': np.max(np.abs(errors))
        }
        
        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.2f} meters")
        
        # Visualize samples
        visualize_results(model, test_loader, config)
        return metrics
    else:
        return avg_loss

def visualize_results(model, test_loader, config, num_samples=3):
    model.eval()
    samples = []
    
    for i, (modalities, depth) in enumerate(test_loader):
        if i >= num_samples:
            break
        samples.append((modalities, depth))
    
    plt.figure(figsize=(15, 5*num_samples))
    for i, (modalities, depth) in enumerate(samples):
        with torch.no_grad():
            modalities = [m.to(config.device) for m in modalities]
            pred = model(modalities).cpu()
        
        # Denormalize
        pred_depth = pred[0].numpy().squeeze() * config.norm_param_depth
        true_depth = depth[0].numpy().squeeze() * config.norm_param_depth
        input_img = modalities[0][0].cpu().numpy().transpose(1, 2, 0)
        input_img = (input_img * 0.5 + 0.5)  # Unnormalize
        
        plt.subplot(num_samples, 3, i*3 + 1)
        plt.imshow(input_img[..., :3])
        plt.title(f'Sample {i+1} Input')
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 2)
        plt.imshow(true_depth, cmap='viridis', vmin=true_depth.min(), vmax=true_depth.max())
        plt.title('Ground Truth')
        plt.colorbar()
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3 + 3)
        plt.imshow(pred_depth, cmap='viridis', vmin=true_depth.min(), vmax=true_depth.max())
        plt.title('Prediction')
        plt.colorbar()
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train()
    
    # Load best model for final evaluation
    model = MultimodalTransformer(config).to(config.device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    test_dataset = MultimodalBathymetryDataset(config.test_ids, config, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    final_metrics = evaluate(model, test_loader, config)
