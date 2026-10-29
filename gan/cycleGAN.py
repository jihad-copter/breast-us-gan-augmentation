import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image, ImageEnhance, ImageFilter
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import random
import warnings
import itertools
import argparse
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CycleGAN Hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETA1 = 0.5
BETA2 = 0.999
EPOCHS = 1000
LAMBDA_CYCLE = 10  # Cycle consistency loss weight
LAMBDA_IDENTITY = 5  # Identity loss weight
CHANNELS = 3

# Residual Block for Generator
class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)

# CycleGAN Generator (ResNet-based)
class CycleGANGenerator(nn.Module):
    """ResNet-based generator for CycleGAN"""
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9):
        super(CycleGANGenerator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Final layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

# CycleGAN Discriminator
class CycleGANDiscriminator(nn.Module):
    """PatchGAN discriminator for CycleGAN"""
    def __init__(self, input_channels=3):
        super(CycleGANDiscriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# Medical Dataset for CycleGAN
class MedicalCycleGANDataset(Dataset):
    """
    Dataset for CycleGAN training with medical images
    Creates domain A (degraded) and domain B (enhanced) versions
    """
    def __init__(self, root_dir, transform=None, domain_type='both', augmentation_style='enhancement', class_filter=None):
        self.root_dir = root_dir
        self.transform = transform
        self.domain_type = domain_type
        self.augmentation_style = augmentation_style
        self.images = []
        self.labels = []
        
        # Collect images from specified classes
        classes_to_include = ['benign', 'malignant'] if class_filter is None else [class_filter]
        
        for class_name in classes_to_include:
            class_path = os.path.join(root_dir, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(class_name)
    
    def create_domain_versions(self, image):
        """Create domain A and B versions of the image"""
        domain_B = image.copy()  # High quality (target domain)
        
        # Create domain A (source domain) based on augmentation style
        if self.augmentation_style == 'enhancement':
            domain_A = self.create_degraded_version(image)
        elif self.augmentation_style == 'denoising':
            domain_A = self.add_realistic_noise(image)
        elif self.augmentation_style == 'contrast':
            domain_A = self.reduce_contrast(image)
        elif self.augmentation_style == 'blur_to_sharp':
            domain_A = self.add_blur(image)
        else:
            domain_A = self.create_style_variation(image)
        
        return domain_A, domain_B
    
    def create_degraded_version(self, image):
        """Create a degraded version for enhancement task"""
        degraded = image.copy()
        
        # Random combination of degradations
        if random.random() > 0.3:
            # Gaussian blur
            blur_radius = random.uniform(0.5, 2.0)
            degraded = degraded.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        if random.random() > 0.4:
            # Reduce sharpness
            enhancer = ImageEnhance.Sharpness(degraded)
            degraded = enhancer.enhance(random.uniform(0.5, 0.8))
        
        if random.random() > 0.5:
            # Add noise
            np_img = np.array(degraded).astype(np.float32)
            noise = np.random.normal(0, random.uniform(8, 15), np_img.shape)
            np_img = np.clip(np_img + noise, 0, 255)
            degraded = Image.fromarray(np_img.astype(np.uint8))
        
        if random.random() > 0.6:
            # Reduce contrast
            enhancer = ImageEnhance.Contrast(degraded)
            degraded = enhancer.enhance(random.uniform(0.7, 0.9))
        
        return degraded
    
    def add_realistic_noise(self, image):
        """Add realistic noise patterns"""
        noisy = np.array(image).astype(np.float32)
        
        # Gaussian noise
        if random.random() > 0.2:
            noise_std = random.uniform(10, 25)
            noise = np.random.normal(0, noise_std, noisy.shape)
            noisy += noise
        
        # Salt and pepper noise
        if random.random() > 0.7:
            noise_prob = random.uniform(0.005, 0.02)
            mask = np.random.random(noisy.shape[:2])
            noisy[mask < noise_prob/2] = 0
            noisy[mask > 1 - noise_prob/2] = 255
        
        # Speckle noise (multiplicative)
        if random.random() > 0.8:
            speckle = np.random.normal(1, 0.1, noisy.shape)
            noisy = noisy * speckle
        
        noisy = np.clip(noisy, 0, 255)
        return Image.fromarray(noisy.astype(np.uint8))
    
    def reduce_contrast(self, image):
        """Reduce image contrast"""
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(0.3, 0.7)
        low_contrast = enhancer.enhance(factor)
        
        # Also reduce brightness slightly
        if random.random() > 0.5:
            brightness_enhancer = ImageEnhance.Brightness(low_contrast)
            low_contrast = brightness_enhancer.enhance(random.uniform(0.8, 0.95))
        
        return low_contrast
    
    def add_blur(self, image):
        """Add various types of blur"""
        blurred = image.copy()
        
        blur_type = random.choice(['gaussian', 'motion', 'defocus'])
        
        if blur_type == 'gaussian':
            radius = random.uniform(1.0, 3.0)
            blurred = blurred.filter(ImageFilter.GaussianBlur(radius=radius))
        elif blur_type == 'motion':
            # Simulate motion blur with directional blur
            radius = random.uniform(0.8, 2.0)
            blurred = blurred.filter(ImageFilter.GaussianBlur(radius=radius))
        else:  # defocus
            radius = random.uniform(1.2, 2.5)
            blurred = blurred.filter(ImageFilter.GaussianBlur(radius=radius))
        
        return blurred
    
    def create_style_variation(self, image):
        """Create style variations"""
        varied = image.copy()
        
        # Color temperature variation
        if random.random() > 0.4:
            enhancer = ImageEnhance.Color(varied)
            varied = enhancer.enhance(random.uniform(0.7, 1.3))
        
        # Brightness variation
        if random.random() > 0.4:
            enhancer = ImageEnhance.Brightness(varied)
            varied = enhancer.enhance(random.uniform(0.8, 1.2))
        
        # Slight contrast change
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(varied)
            varied = enhancer.enhance(random.uniform(0.9, 1.1))
        
        return varied
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Create domain versions
        domain_A, domain_B = self.create_domain_versions(image)
        
        # Apply transforms
        if self.transform:
            domain_A = self.transform(domain_A)
            domain_B = self.transform(domain_B)
        
        if self.domain_type == 'A':
            return domain_A, self.labels[idx]
        elif self.domain_type == 'B':
            return domain_B, self.labels[idx]
        else:
            return domain_A, domain_B, self.labels[idx]

# Replay buffer for stable training
class ReplayBuffer:
    """Buffer to store previously generated images"""
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

# Loss functions
class CycleGANLoss(nn.Module):
    """Combined losses for CycleGAN"""
    def __init__(self, lambda_cycle=10, lambda_identity=5):
        super(CycleGANLoss, self).__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def adversarial_loss(self, prediction, target_is_real):
        """Calculate adversarial loss"""
        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)
        return self.mse_loss(prediction, target)
    
    def cycle_consistency_loss(self, real_A, cycled_A, real_B, cycled_B):
        """Calculate cycle consistency loss"""
        loss_A = self.l1_loss(cycled_A, real_A)
        loss_B = self.l1_loss(cycled_B, real_B)
        return (loss_A + loss_B) * self.lambda_cycle
    
    def identity_loss(self, real_A, same_A, real_B, same_B):
        """Calculate identity loss"""
        loss_A = self.l1_loss(same_A, real_A)
        loss_B = self.l1_loss(same_B, real_B)
        return (loss_A + loss_B) * self.lambda_identity

# Training function
def train_cyclegan(train_directory, epochs=EPOCHS, save_interval=25, 
                   augmentation_style='enhancement', class_specific=True):
    """Train CycleGAN for medical image translation"""
    
    print(f"\n{'='*60}")
    print(f"Training CycleGAN for medical image translation")
    print(f"Augmentation style: {augmentation_style}")
    print(f"{'='*60}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    models = {}
    classes_to_train = ['benign', 'malignant'] if class_specific else [None]
    
    for class_name in classes_to_train:
        print(f"\n--- Training for {class_name or 'all classes'} ---")
        
        # Create dataset
        dataset = MedicalCycleGANDataset(
            root_dir=train_directory,
            transform=transform,
            domain_type='both',
            augmentation_style=augmentation_style,
            class_filter=class_name
        )
        
        if len(dataset) == 0:
            print(f"No images found for {class_name or 'all classes'}")
            continue
        
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=2, pin_memory=True, drop_last=True)
        
        print(f"Dataset size: {len(dataset)} images")
        print(f"Batches per epoch: {len(dataloader)}")
        
        # Initialize networks
        G_AB = CycleGANGenerator(CHANNELS, CHANNELS).to(device)  # A -> B (degraded -> enhanced)
        G_BA = CycleGANGenerator(CHANNELS, CHANNELS).to(device)  # B -> A (enhanced -> degraded)
        D_A = CycleGANDiscriminator(CHANNELS).to(device)         # Discriminator for domain A
        D_B = CycleGANDiscriminator(CHANNELS).to(device)         # Discriminator for domain B
        
        # Test forward pass
        try:
            test_batch_size = min(4, BATCH_SIZE)
            test_A = torch.randn(test_batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            test_B = torch.randn(test_batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            
            # Set to eval mode for testing
            G_AB.eval()
            G_BA.eval()
            D_A.eval()
            D_B.eval()
            
            with torch.no_grad():
                fake_B = G_AB(test_A)
                fake_A = G_BA(test_B)
                disc_A_result = D_A(test_A)
                disc_B_result = D_B(test_B)
            
            # Set back to train mode
            G_AB.train()
            G_BA.train()
            D_A.train()
            D_B.train()
            
            print(f"Network test successful:")
            print(f"  G_AB output: {fake_B.shape}")
            print(f"  G_BA output: {fake_A.shape}")
            print(f"  D_A output: {disc_A_result.shape}")
            print(f"  D_B output: {disc_B_result.shape}")
            
        except Exception as e:
            print(f"Network test failed: {e}")
            continue
        
        # Optimizers
        optimizer_G = optim.Adam(
            itertools.chain(G_AB.parameters(), G_BA.parameters()),
            lr=LEARNING_RATE_G, betas=(BETA1, BETA2)
        )
        optimizer_D_A = optim.Adam(D_A.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))
        optimizer_D_B = optim.Adam(D_B.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))
        
        # Learning rate schedulers
        lr_scheduler_G = optim.lr_scheduler.LambdaLR(
            optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - epochs//2) / (epochs//2)
        )
        lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(
            optimizer_D_A, lr_lambda=lambda epoch: 1.0 - max(0, epoch - epochs//2) / (epochs//2)
        )
        lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(
            optimizer_D_B, lr_lambda=lambda epoch: 1.0 - max(0, epoch - epochs//2) / (epochs//2)
        )
        
        # Loss function
        criterion = CycleGANLoss(lambda_cycle=LAMBDA_CYCLE, lambda_identity=LAMBDA_IDENTITY)
        
        # Replay buffers
        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()
        
        # Training metrics
        g_losses = []
        d_a_losses = []
        d_b_losses = []
        cycle_losses = []
        
        # Fixed samples for monitoring
        fixed_samples = None
        for batch_idx, batch_data in enumerate(dataloader):
            if len(batch_data) == 3:  # domain_A, domain_B, labels
                real_A, real_B, _ = batch_data
                fixed_samples = (real_A[:4].to(device), real_B[:4].to(device))
                break
        
        # Create directories
        class_suffix = f"_{class_name}_{augmentation_style}" if class_name else f"_all_{augmentation_style}"
        os.makedirs(f"cyclegan_models{class_suffix}", exist_ok=True)
        os.makedirs(f"cyclegan_samples{class_suffix}", exist_ok=True)
        
        print(f"Starting CycleGAN training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_a_loss = 0
            epoch_d_b_loss = 0
            epoch_cycle_loss = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, batch_data in enumerate(progress_bar):
                if len(batch_data) == 3:
                    real_A, real_B, _ = batch_data
                else:
                    continue
                
                real_A = real_A.to(device)
                real_B = real_B.to(device)
                batch_size = real_A.size(0)
                
                # ===============================
                # Train Generators G_AB and G_BA
                # ===============================
                optimizer_G.zero_grad()
                
                # Identity loss
                same_B = G_AB(real_B)
                same_A = G_BA(real_A)
                identity_loss = criterion.identity_loss(real_A, same_A, real_B, same_B)
                
                # Adversarial loss
                fake_B = G_AB(real_A)
                pred_fake_B = D_B(fake_B)
                loss_GAN_AB = criterion.adversarial_loss(pred_fake_B, True)
                
                fake_A = G_BA(real_B)
                pred_fake_A = D_A(fake_A)
                loss_GAN_BA = criterion.adversarial_loss(pred_fake_A, True)
                
                # Cycle consistency loss
                recovered_A = G_BA(fake_B)
                recovered_B = G_AB(fake_A)
                cycle_loss = criterion.cycle_consistency_loss(real_A, recovered_A, real_B, recovered_B)
                
                # Total generator loss
                loss_G = identity_loss + loss_GAN_AB + loss_GAN_BA + cycle_loss
                loss_G.backward()
                optimizer_G.step()
                
                # ===============================
                # Train Discriminator D_A
                # ===============================
                optimizer_D_A.zero_grad()
                
                # Real loss
                pred_real_A = D_A(real_A)
                loss_D_real_A = criterion.adversarial_loss(pred_real_A, True)
                
                # Fake loss
                fake_A_buffered = fake_A_buffer.push_and_pop(fake_A)
                pred_fake_A = D_A(fake_A_buffered.detach())
                loss_D_fake_A = criterion.adversarial_loss(pred_fake_A, False)
                
                # Combined loss
                loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
                loss_D_A.backward()
                optimizer_D_A.step()
                
                # ===============================
                # Train Discriminator D_B
                # ===============================
                optimizer_D_B.zero_grad()
                
                # Real loss
                pred_real_B = D_B(real_B)
                loss_D_real_B = criterion.adversarial_loss(pred_real_B, True)
                
                # Fake loss
                fake_B_buffered = fake_B_buffer.push_and_pop(fake_B)
                pred_fake_B = D_B(fake_B_buffered.detach())
                loss_D_fake_B = criterion.adversarial_loss(pred_fake_B, False)
                
                # Combined loss
                loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
                loss_D_B.backward()
                optimizer_D_B.step()
                
                # Update metrics
                epoch_g_loss += loss_G.item()
                epoch_d_a_loss += loss_D_A.item()
                epoch_d_b_loss += loss_D_B.item()
                epoch_cycle_loss += cycle_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'G_Loss': f'{loss_G.item():.3f}',
                    'D_A_Loss': f'{loss_D_A.item():.3f}',
                    'D_B_Loss': f'{loss_D_B.item():.3f}',
                    'Cycle': f'{cycle_loss.item():.3f}'
                })
            
            # Calculate averages
            num_batches = max(len(dataloader), 1)
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_a_loss = epoch_d_a_loss / num_batches
            avg_d_b_loss = epoch_d_b_loss / num_batches
            avg_cycle_loss = epoch_cycle_loss / num_batches
            
            g_losses.append(avg_g_loss)
            d_a_losses.append(avg_d_a_loss)
            d_b_losses.append(avg_d_b_loss)
            cycle_losses.append(avg_cycle_loss)
            
            # Update learning rates after half the epochs
            if epoch > epochs // 2:
                lr_scheduler_G.step()
                lr_scheduler_D_A.step()
                lr_scheduler_D_B.step()
            
            # Save samples
            if (epoch + 1) % save_interval == 0:
                G_AB.eval()
                G_BA.eval()
                with torch.no_grad():
                    if fixed_samples is not None:
                        fixed_real_A, fixed_real_B = fixed_samples
                        fake_B_samples = G_AB(fixed_real_A)
                        fake_A_samples = G_BA(fixed_real_B)
                        
                        # Cycle translations
                        cycle_A_samples = G_BA(fake_B_samples)
                        cycle_B_samples = G_AB(fake_A_samples)
                        
                        # Denormalize for visualization
                        real_A_display = (fixed_real_A + 1) / 2
                        real_B_display = (fixed_real_B + 1) / 2
                        fake_B_display = (fake_B_samples + 1) / 2
                        fake_A_display = (fake_A_samples + 1) / 2
                        cycle_A_display = (cycle_A_samples + 1) / 2
                        cycle_B_display = (cycle_B_samples + 1) / 2
                        
                        # Create comparison grid: Real A | Fake B | Cycle A | Real B | Fake A | Cycle B
                        comparison = torch.cat([
                            real_A_display,      # Original domain A
                            fake_B_display,      # A -> B translation
                            cycle_A_display,     # A -> B -> A cycle
                            real_B_display,      # Original domain B
                            fake_A_display,      # B -> A translation
                            cycle_B_display      # B -> A -> B cycle
                        ], dim=0)
                        
                        vutils.save_image(comparison, 
                                        f"cyclegan_samples{class_suffix}/epoch_{epoch+1}.png", 
                                        nrow=4, normalize=False, padding=2)
                G_AB.train()
                G_BA.train()
            
            # Progress report
            if (epoch + 1) % 25 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Generator Loss: {avg_g_loss:.4f}")
                print(f"Discriminator A Loss: {avg_d_a_loss:.4f}")
                print(f"Discriminator B Loss: {avg_d_b_loss:.4f}")
                print(f"Cycle Loss: {avg_cycle_loss:.4f}")
        
        # Save final models
        torch.save(G_AB.state_dict(), f"cyclegan_models{class_suffix}/G_AB.pth")
        torch.save(G_BA.state_dict(), f"cyclegan_models{class_suffix}/G_BA.pth")
        torch.save(D_A.state_dict(), f"cyclegan_models{class_suffix}/D_A.pth")
        torch.save(D_B.state_dict(), f"cyclegan_models{class_suffix}/D_B.pth")
        
        # Plot training curves
        if len(g_losses) > 0:
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(g_losses, label='Generator', alpha=0.7)
            plt.plot(d_a_losses, label='Discriminator A', alpha=0.7)
            plt.plot(d_b_losses, label='Discriminator B', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'CycleGAN Losses ({class_name or "all"})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 2)
            plt.plot(cycle_losses, label='Cycle Consistency', color='green', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Cycle Loss')
            plt.title('Cycle Consistency Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 3)
            # Discriminator balance
            avg_d_loss = [(a + b) / 2 for a, b in zip(d_a_losses, d_b_losses)]
            # Discriminator balance
            avg_d_loss = [(a + b) / 2 for a, b in zip(d_a_losses, d_b_losses)]
            balance = np.array(g_losses) / (np.array(avg_d_loss) + 1e-8)
            plt.plot(balance, label='G/D Ratio', color='purple')
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
            plt.xlabel('Epoch')
            plt.ylabel('Loss Ratio')
            plt.title('Training Balance')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            # Loss components over time
            plt.plot(g_losses, label='Total Generator', alpha=0.7)
            plt.plot(cycle_losses, label='Cycle Consistency', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Generator Loss Components')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'cyclegan_curves{class_suffix}.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        models[class_name or 'all'] = {
            'G_AB': G_AB,
            'G_BA': G_BA,
            'D_A': D_A,
            'D_B': D_B,
            'class_suffix': class_suffix
        }
        
        print(f"CycleGAN training completed for {class_name or 'all classes'}")
    
    return models

# Generate augmented dataset using CycleGAN
def generate_cyclegan_dataset(models, train_directory, output_directory, 
                             augmentation_style='enhancement', images_per_original=10):
    """Generate augmented dataset using CycleGAN translation"""
    
    print("=== Generating CycleGAN Augmented Dataset ===")
    print(f"Available models: {list(models.keys())}")
    
    # Create output directories
    for class_name in ['benign', 'malignant']:
        os.makedirs(os.path.join(output_directory, class_name), exist_ok=True)
    
    total_generated = 0
    generation_stats = {'benign': 0, 'malignant': 0}
    
    # Setup transforms
    transform_for_generation = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    for class_name in ['benign', 'malignant']:
        print(f"\n--- Processing {class_name} class ---")
        class_path = os.path.join(train_directory, class_name)
        
        if not os.path.exists(class_path):
            print(f"Warning: {class_name} directory not found at {class_path}")
            continue
        
        # Get appropriate models
        G_AB = None  # Domain A -> B (degraded -> enhanced)
        G_BA = None  # Domain B -> A (enhanced -> degraded)
        
        if class_name in models:
            G_AB = models[class_name]['G_AB']
            G_BA = models[class_name]['G_BA']
            print(f"Using class-specific models for {class_name}")
        elif 'all' in models:
            G_AB = models['all']['G_AB']
            G_BA = models['all']['G_BA']
            print(f"Using 'all' models for {class_name}")
        elif len(models) > 0:
            first_model_key = list(models.keys())[0]
            G_AB = models[first_model_key]['G_AB']
            G_BA = models[first_model_key]['G_BA']
            print(f"Using available models '{first_model_key}' for {class_name}")
        
        if G_AB is None or G_BA is None:
            print(f"ERROR: No models available for {class_name} class")
            continue
        
        G_AB.eval()
        G_BA.eval()
        
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        print(f"Found {len(image_files)} {class_name} images to process")
        
        if len(image_files) == 0:
            print(f"Warning: No valid image files found in {class_path}")
            continue
        
        successful_generations = 0
        failed_generations = 0
        
        for img_idx, img_name in enumerate(tqdm(image_files, desc=f"Generating {class_name}")):
            original_path = os.path.join(class_path, img_name)
            base_name = os.path.splitext(img_name)[0]
            
            try:
                # Load original image
                original_img = Image.open(original_path).convert('RGB')
                
                # Save original
                orig_output = os.path.join(output_directory, class_name, f"orig_{base_name}.png")
                original_img.save(orig_output)
                
                with torch.no_grad():
                    # Generate multiple variations using both directions
                    for i in range(images_per_original):
                        try:
                            # Create dataset helper for domain creation
                            dataset_temp = MedicalCycleGANDataset(
                                train_directory, 
                                augmentation_style=augmentation_style
                            )
                            
                            # Create domain A and B versions
                            domain_A, domain_B = dataset_temp.create_domain_versions(original_img)
                            
                            # Use different translation strategies
                            if i % 3 == 0:
                                # A -> B translation (degraded -> enhanced)
                                input_img = domain_A
                                input_tensor = transform_for_generation(input_img).unsqueeze(0).to(device)
                                generated_tensor = G_AB(input_tensor)
                                translation_type = "A2B"
                            elif i % 3 == 1:
                                # B -> A translation (enhanced -> degraded) then back to B
                                input_img = domain_B
                                input_tensor = transform_for_generation(input_img).unsqueeze(0).to(device)
                                # First B -> A, then A -> B for round-trip
                                intermediate = G_BA(input_tensor)
                                generated_tensor = G_AB(intermediate)
                                translation_type = "B2A2B"
                            else:
                                # Direct B -> A translation for variety
                                input_img = domain_B
                                input_tensor = transform_for_generation(input_img).unsqueeze(0).to(device)
                                generated_tensor = G_BA(input_tensor)
                                translation_type = "B2A"
                            
                            # Convert back to PIL
                            generated_tensor = (generated_tensor + 1) / 2  # Denormalize
                            generated_tensor = torch.clamp(generated_tensor, 0, 1)
                            
                            generated_img = transforms.ToPILImage()(generated_tensor.squeeze(0).cpu())
                            
                            # Save generated image
                            gen_name = f"cyclegan_{augmentation_style}_{translation_type}_{base_name}_{i+1:02d}.png"
                            gen_path = os.path.join(output_directory, class_name, gen_name)
                            generated_img.save(gen_path)
                            
                            total_generated += 1
                            generation_stats[class_name] += 1
                            successful_generations += 1
                            
                            # Save input examples (first few iterations)
                            if i < 2:
                                input_name = f"input_{translation_type}_{augmentation_style}_{base_name}_{i}.png"
                                input_path = os.path.join(output_directory, class_name, input_name)
                                input_img.save(input_path)
                        
                        except Exception as inner_e:
                            print(f"Failed to generate variation {i+1} for {img_name}: {str(inner_e)}")
                            failed_generations += 1
                            continue
                
                # Progress report
                if (img_idx + 1) % 10 == 0:
                    print(f"Progress for {class_name}: {img_idx + 1}/{len(image_files)} images processed")
                    print(f"  Successful generations: {successful_generations}")
                    print(f"  Failed generations: {failed_generations}")
                
            except Exception as e:
                print(f"Failed to process {img_name} in {class_name}: {str(e)}")
                failed_generations += 1
                continue
        
        print(f"\nCompleted {class_name} class:")
        print(f"  Total images processed: {len(image_files)}")
        print(f"  Successful generations: {successful_generations}")
        print(f"  Failed generations: {failed_generations}")
        print(f"  Images added to dataset: {generation_stats[class_name]}")
        
        G_AB.train()
        G_BA.train()
    
    # Print statistics
    print(f"\n=== CycleGAN Dataset Generation Complete ===")
    print(f"Total generated images: {total_generated}")
    for class_name, count in generation_stats.items():
        if os.path.exists(os.path.join(train_directory, class_name)):
            original_count = len([f for f in os.listdir(os.path.join(train_directory, class_name))
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            print(f"{class_name}: {original_count} original + {count} generated = {original_count + count} total")
            print(f"  Augmentation factor: {count/original_count:.1f}x")
    
    return generation_stats

# Evaluation functions
def evaluate_cyclegan_quality(models, test_samples=30):
    """Evaluate CycleGAN model quality"""
    
    print("\n=== CycleGAN Quality Evaluation ===")
    
    for model_name, model_info in models.items():
        G_AB = model_info['G_AB']
        G_BA = model_info['G_BA']
        
        G_AB.eval()
        G_BA.eval()
        
        print(f"\nEvaluating {model_name} models...")
        
        with torch.no_grad():
            # Create test samples
            test_A = torch.randn(test_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            test_B = torch.randn(test_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            
            # Test translations
            fake_B = G_AB(test_A)  # A -> B
            fake_A = G_BA(test_B)  # B -> A
            
            # Test cycle consistency
            recovered_A = G_BA(fake_B)  # A -> B -> A
            recovered_B = G_AB(fake_A)  # B -> A -> B
            
            # Calculate cycle consistency
            cycle_A_loss = F.l1_loss(recovered_A, test_A).item()
            cycle_B_loss = F.l1_loss(recovered_B, test_B).item()
            
            print(f"  Cycle consistency A->B->A: {cycle_A_loss:.4f}")
            print(f"  Cycle consistency B->A->B: {cycle_B_loss:.4f}")
            print(f"  Average cycle consistency: {(cycle_A_loss + cycle_B_loss) / 2:.4f}")
            
            # Test diversity
            pairwise_distances_AB = []
            pairwise_distances_BA = []
            sample_size = min(10, test_samples)
            
            for i in range(sample_size):
                for j in range(i+1, sample_size):
                    dist_AB = F.mse_loss(fake_B[i], fake_B[j]).item()
                    dist_BA = F.mse_loss(fake_A[i], fake_A[j]).item()
                    pairwise_distances_AB.append(dist_AB)
                    pairwise_distances_BA.append(dist_BA)
            
            avg_diversity_AB = np.mean(pairwise_distances_AB)
            avg_diversity_BA = np.mean(pairwise_distances_BA)
            
            print(f"  A->B translation diversity: {avg_diversity_AB:.4f}")
            print(f"  B->A translation diversity: {avg_diversity_BA:.4f}")
            
            diversity_status_AB = "High" if avg_diversity_AB > 0.1 else "Medium" if avg_diversity_AB > 0.05 else "Low"
            diversity_status_BA = "High" if avg_diversity_BA > 0.1 else "Medium" if avg_diversity_BA > 0.05 else "Low"
            
            print(f"  A->B diversity assessment: {diversity_status_AB}")
            print(f"  B->A diversity assessment: {diversity_status_BA}")
        
        G_AB.train()
        G_BA.train()

def create_cyclegan_demo(models, output_path="cyclegan_demonstrations"):
    """Create CycleGAN demonstration images"""
    
    print("\n=== Creating CycleGAN Demonstration ===")
    os.makedirs(output_path, exist_ok=True)
    
    for model_name, model_info in models.items():
        G_AB = model_info['G_AB']
        G_BA = model_info['G_BA']
        
        G_AB.eval()
        G_BA.eval()
        
        with torch.no_grad():
            # Create test inputs
            test_inputs = []
            
            # Random inputs
            random_A = torch.randn(4, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            random_B = torch.randn(4, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            test_inputs.append(("Random_A", random_A, "A2B"))
            test_inputs.append(("Random_B", random_B, "B2A"))
            
            # Pattern inputs
            pattern_A = torch.zeros(4, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            pattern_B = torch.ones(4, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device) * 0.5
            for i in range(4):
                pattern_A[i, :, 64:192, 64:192] = 1.0
                pattern_B[i, :, 32:96, 32:96] = -1.0
                pattern_B[i, :, 160:224, 160:224] = 1.0
            
            test_inputs.append(("Pattern_A", pattern_A, "A2B"))
            test_inputs.append(("Pattern_B", pattern_B, "B2A"))
            
            for demo_name, input_tensor, direction in test_inputs:
                if direction == "A2B":
                    output_tensor = G_AB(input_tensor)
                else:
                    output_tensor = G_BA(input_tensor)
                
                # Cycle translation
                if direction == "A2B":
                    cycle_tensor = G_BA(output_tensor)
                else:
                    cycle_tensor = G_AB(output_tensor)
                
                # Denormalize for display
                input_display = (input_tensor + 1) / 2
                output_display = (output_tensor + 1) / 2
                cycle_display = (cycle_tensor + 1) / 2
                
                # Create comparison: Input | Translation | Cycle
                comparison = torch.cat([input_display, output_display, cycle_display], dim=3)
                
                vutils.save_image(comparison, 
                                f"{output_path}/{model_name}_{demo_name}_{direction}.png",
                                nrow=2, normalize=False, padding=2)
        
        G_AB.train()
        G_BA.train()
    
    print(f"Demo images saved to {output_path}/")

def validate_dataset_cyclegan(directory):
    """Validate dataset for CycleGAN training"""
    if not os.path.exists(directory):
        return False, f"Directory not found: {directory}"
    
    total_images = 0
    class_counts = {}
    
    for class_name in ['benign', 'malignant']:
        class_path = os.path.join(directory, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            class_counts[class_name] = count
            total_images += count
        else:
            class_counts[class_name] = 0
    
    return total_images > 0, class_counts

def get_cyclegan_configuration():
    """Get CycleGAN configuration"""
    config = {
        'train_directory': r"F:\jihad\Breast GAN types\breast\breast\split_dataset\train",
        'output_directory': r"F:\jihad\Breast GAN types\breast\breast\cyclegan_augmented",
        'augmentation_style': 'enhancement',
        'epochs': 1000,
        'class_specific': True,
        'images_per_original': 10,
        'save_interval': 10
    }
    
    return config

def main():
    """Main function for CycleGAN training and evaluation"""
    
    print("=" * 80)
    print("CYCLEGAN FOR MEDICAL IMAGE TRANSLATION")
    print("Unpaired image-to-image translation for medical imaging")
    print("=" * 80)
    
    # Get configuration
    config = get_cyclegan_configuration()
    
    print(f"\n=== Configuration ===")
    print(f"Device: {device}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {config['epochs']}")
    print(f"Augmentation style: {config['augmentation_style']}")
    print(f"Training directory: {config['train_directory']}")
    print(f"Output directory: {config['output_directory']}")
    print(f"Images per original: {config['images_per_original']}")
    
    # Validate dataset
    print(f"\n=== Dataset Validation ===")
    is_valid, class_info = validate_dataset_cyclegan(config['train_directory'])
    
    if not is_valid:
        print(f"Dataset validation failed: {class_info}")
        print("Please check your dataset path and structure.")
        return False
    
    print("Dataset structure:")
    total_images = 0
    for class_name, count in class_info.items():
        print(f"  {class_name}: {count} images")
        total_images += count
    print(f"Total images: {total_images}")
    
    if total_images < 20:
        print("Warning: CycleGAN typically needs more images for stable training.")
        print("However, we'll proceed with the available data.")
    
    # Show available augmentation styles
    print(f"\n=== Augmentation Styles ===")
    style_info = {
        'enhancement': 'Degraded ↔ Enhanced image translation',
        'denoising': 'Noisy ↔ Clean image translation',
        'contrast': 'Low contrast ↔ High contrast translation',
        'blur_to_sharp': 'Blurry ↔ Sharp image translation'
    }
    
    for style, description in style_info.items():
        marker = " [SELECTED]" if style == config['augmentation_style'] else ""
        print(f"  {style}: {description}{marker}")
    
    # Training phase
    print(f"\n=== Training Phase ===")
    print("Starting CycleGAN training...")
    
    try:
        trained_models = train_cyclegan(
            train_directory=config['train_directory'],
            epochs=config['epochs'],
            save_interval=config['save_interval'],
            augmentation_style=config['augmentation_style'],
            class_specific=config['class_specific']
        )
        
        if not trained_models:
            print("ERROR: No models were successfully trained")
            print("Common issues:")
            print("- Insufficient GPU memory (try reducing BATCH_SIZE)")
            print("- Dataset path issues")
            print("- Image format compatibility")
            print("- Need more diverse training images")
            return False
        
        print(f"SUCCESS: Training completed for {len(trained_models)} model set(s)")
        
    except Exception as e:
        print(f"TRAINING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Quality evaluation
    print(f"\n=== Quality Evaluation ===")
    try:
        evaluate_cyclegan_quality(trained_models, test_samples=20)
    except Exception as e:
        print(f"Quality evaluation failed: {str(e)}")
    
    # Create demonstrations
    print(f"\n=== Creating Demonstrations ===")
    try:
        create_cyclegan_demo(trained_models, output_path="cyclegan_demonstrations")
        print("Demonstration images created successfully")
    except Exception as e:
        print(f"Demo creation failed: {str(e)}")
    
    # Generate augmented dataset
    print(f"\n=== Dataset Generation ===")
    try:
        generation_stats = generate_cyclegan_dataset(
            models=trained_models,
            train_directory=config['train_directory'],
            output_directory=config['output_directory'],
            augmentation_style=config['augmentation_style'],
            images_per_original=config['images_per_original']
        )
        
        print("Augmented dataset generated successfully")
        
    except Exception as e:
        print(f"Dataset generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Results summary
    print(f"\n=== CycleGAN Results Summary ===")
    print("Generated files and directories:")
    
    file_info = [
        ("cyclegan_models_*/", "Trained generator and discriminator models (G_AB, G_BA, D_A, D_B)"),
        ("cyclegan_samples_*/", "Training progress images showing translations"),
        ("cyclegan_curves_*.png", "Training loss curves and metrics"),
        ("cyclegan_demonstrations/", "Model capability demonstrations"),
        (config['output_directory'], "Augmented dataset for classification")
    ]
    
    for filename, description in file_info:
        print(f"  {filename:<35} - {description}")
    
    print(f"\n=== CycleGAN Advantages for Medical Imaging ===")
    advantages = [
        "Unpaired training - no need for matched image pairs",
        "Bidirectional translation capabilities (A↔B)",
        "Cycle consistency preserves anatomical structure",
        "Identity loss maintains class-specific features",
        "Handles domain shift between imaging conditions",
        "More flexible than supervised methods",
        "Creates diverse augmentations through multiple translation paths"
    ]
    
    for advantage in advantages:
        print(f"  • {advantage}")
    
    print(f"\n=== Expected Dataset Expansion ===")
    print("Your augmented dataset now includes:")
    expected_benign = config['images_per_original'] * class_info['benign'] if class_info['benign'] > 0 else 0
    expected_malignant = config['images_per_original'] * class_info['malignant'] if class_info['malignant'] > 0 else 0
    
    print(f"  • Benign: {class_info['benign']} original + {expected_benign} generated = {class_info['benign'] + expected_benign} total")
    print(f"  • Malignant: {class_info['malignant']} original + {expected_malignant} generated = {class_info['malignant'] + expected_malignant} total")
    print(f"  • Total dataset: {total_images + expected_benign + expected_malignant} images")
    print(f"  • Augmentation factor: {config['images_per_original']}x per original image")
    
    print(f"\n=== Translation Types Generated ===")
    print("For each original image, CycleGAN creates:")
    print("  • A→B translations (degraded → enhanced)")
    print("  • B→A→B round-trip translations (enhanced → degraded → enhanced)")  
    print("  • B→A translations (enhanced → degraded)")
    print("  • Cycle-consistent variations preserving anatomical features")
    
    print(f"\n=== Next Steps ===")
    next_steps = [
        "Review training samples showing bidirectional translations",
        "Examine cycle consistency in demonstration images", 
        "Compare CycleGAN vs original dataset classification performance",
        "Test with different augmentation styles (denoising, contrast, blur)",
        "Analyze generated image quality and diversity",
        "Use augmented dataset for training medical image classifiers",
        "Fine-tune cycle and identity loss weights if needed"
    ]
    
    for i, step in enumerate(next_steps, 1):
        print(f"  {i}. {step}")
    
    print(f"\n=== Training Tips for Better Results ===")
    tips = [
        "Increase epochs (200+) for better convergence on larger datasets",
        "Adjust LAMBDA_CYCLE (10) and LAMBDA_IDENTITY (5) based on results",
        "Monitor cycle consistency loss - should decrease over time",
        "Check discriminator balance - G/D ratio should stay around 1.0",
        "Use more diverse original images for better domain coverage"
    ]
    
    for tip in tips:
        print(f"  • {tip}")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 80)
        print("CYCLEGAN PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nYour CycleGAN models are ready for unpaired medical image translation!")
        print("The augmented dataset provides diverse, cycle-consistent variations!")
        print("Each original image now has 10 realistic augmented versions!")
    else:
        print("\n" + "=" * 80)
        print("CYCLEGAN PIPELINE ENCOUNTERED ERRORS")
        print("=" * 80)
        print("\nPlease check the error messages above and:")
        print("1. Verify dataset paths and structure")
        print("2. Ensure sufficient GPU memory and training data")
        print("3. Check image file formats and accessibility")
        print("4. Consider adjusting hyperparameters for your dataset")
        print("5. Try reducing BATCH_SIZE if encountering memory issues")