import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler

# Parameters
dataset = "spot6"
norm_param = np.load('/kaggle/input/magicbethynet/MagicBathyNet/agia_napa/norm_param_spot6_an.npy')
WINDOW_SIZE = (30, 30)
STRIDE = 2
BATCH_SIZE = 1
MAIN_FOLDER = '/kaggle/input/magicbethynet/MagicBathyNet/agia_napa'
DATA_FOLDER = MAIN_FOLDER + '/img/spot6/img_{}.tif'
train_images = ['409', '418', '350', '399', '361', '430', '380', '359', '371', '377', '379', '360', '368', '419', '389', '420', '401', '408', '352', '388', '362', '421', '412', '351', '349', '390', '400', '378']
test_images = ['411', '387', '410', '398', '370', '369', '397']

# Define the Self-Supervised Neural Network
class SelfSupervisedNN(nn.Module):
    def __init__(self, in_channels):
        super(SelfSupervisedNN, self).__init__()
        self.n_channels = in_channels

        # Feature extraction backbone
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Output size becomes (64, 15, 15)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # Output size becomes (256, 7, 7)
        )

        # Decoder for image reconstruction
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Output size becomes (128, 14, 14)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # Output size becomes (64, 28, 28)
            nn.ReLU(inplace=True),
            nn.Upsample(size=(30, 30)),  # Upsample to match input size
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),  # Final convolution to match input channels
            nn.Sigmoid()  # Output is in the range [0, 1]
        )

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        # Reconstruct the input image
        reconstructed = self.decoder(features)
        return reconstructed


# Define the reconstruction loss (MSE for self-supervised learning)
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, output, target):
        return F.mse_loss(output, target)

# Dataset class (modified for self-supervised learning)
class SelfSupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, cache=False, augmentation=True):
        super(SelfSupervisedDataset, self).__init__()
        self.augmentation = augmentation
        self.cache = cache
        self.data_files = [data_files.format(id) for id in ids]
        self.data_cache_ = {}

    def __len__(self):
        return 10000  # Default epoch size

    def __getitem__(self, i):
        random_idx = random.randint(0, len(self.data_files) - 1)

        if random_idx in self.data_cache_:
            data = self.data_cache_[random_idx]
        else:
            data = np.asarray(io.imread(self.data_files[random_idx]).transpose((2, 0, 1)), dtype='float32')
            data = (data - norm_param[0][:, np.newaxis, np.newaxis]) / (norm_param[1][:, np.newaxis, np.newaxis] - norm_param[0][:, np.newaxis, np.newaxis])
            if self.cache:
                self.data_cache_[random_idx] = data

        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]

        if self.augmentation:
            data_p = self.data_augmentation(data_p)

        return torch.from_numpy(data_p), torch.from_numpy(data_p)  # Input and target are the same

    @staticmethod
    def data_augmentation(array, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        if will_flip:
            array = array[::-1, :] if len(array.shape) == 2 else array[:, ::-1, :]
        if will_mirror:
            array = array[:, ::-1] if len(array.shape) == 2 else array[:, :, ::-1]
        return np.copy(array)

# Helper function to get random positions
def get_random_pos(img, window_shape):
    """Extract a 2D random patch of shape window_shape in the image."""
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w)
    x2 = x1 + w
    y1 = random.randint(0, H - h)
    y2 = y1 + h
    return x1, x2, y1, y2

# Load the network on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = SelfSupervisedNN(3).to(device)

# Load the datasets
train_set = SelfSupervisedDataset(train_images, DATA_FOLDER, cache=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

# Define the optimizer
base_lr = 0.0001
optimizer = optim.Adam(net.parameters(), lr=base_lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10], gamma=0.1)

# Training function
def train(net, optimizer, epochs, scheduler=None, save_epoch=15):
    criterion = ReconstructionLoss()
    iter_ = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.to(device).float()), Variable(target.to(device).float())
            optimizer.zero_grad()

            # Forward pass
            output = net(data)

            # Compute loss
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if iter_ % 100 == 0:
                print(f'Epoch {e}/{epochs}, Iter {iter_}, Loss: {loss.item()}')

            iter_ += 1

        if e % save_epoch == 0:
            torch.save(net.state_dict(), f'model_epoch{e}.pth')

    torch.save(net.state_dict(), 'model_final.pth')


# Train the network
train(net, optimizer, 50, scheduler)

# Testing function (optional, for visualization)
def test(net, test_ids):
    net.eval()
    all_preds = []

    for id_ in test_ids:
        img = np.asarray(io.imread(DATA_FOLDER.format(id_)).transpose((2, 0, 1)), dtype='float32')
        img = (img - norm_param[0][:, np.newaxis, np.newaxis]) / (norm_param[1][:, np.newaxis, np.newaxis] - norm_param[0][:, np.newaxis, np.newaxis])
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = net(img_tensor.float()).cpu().numpy().squeeze()

        all_preds.append(pred)

    return all_preds

# Test the network
all_preds = test(net, test_images)

# Save the results
for pred, id_ in zip(all_preds, test_images):
    plt.imshow(pred.transpose(1, 2, 0), cmap='viridis')
    plt.show()
    io.imsave(f'reconstruction_{id_}.tif', pred.transpose(1, 2, 0))
