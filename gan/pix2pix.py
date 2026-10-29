import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image, ImageEnhance
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import random
import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Pix2Pix Hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0002
BETA1 = 0.5
BETA2 = 0.999
EPOCHS = 1000
LAMBDA_L1 = 100  # L1 loss weight for reconstruction
CHANNELS = 3

# U-Net Generator for Pix2Pix - FIXED VERSION
class UNetBlock(nn.Module):
    """Basic U-Net block with skip connections - Fixed for BatchNorm issues"""
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False, use_batch_norm=True):
        super(UNetBlock, self).__init__()
        self.down = down
        
        if down:
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)]
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(0.5) if use_dropout else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x

class Pix2PixGenerator(nn.Module):
    """U-Net based generator for Pix2Pix - FIXED VERSION"""
    def __init__(self, input_channels=3, output_channels=3):
        super(Pix2PixGenerator, self).__init__()
        
        # Encoder (Contracting path)
        self.down1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),  # No BatchNorm in first layer
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = UNetBlock(64, 128, down=True)            # 128x128 -> 64x64
        self.down3 = UNetBlock(128, 256, down=True)           # 64x64 -> 32x32
        self.down4 = UNetBlock(256, 512, down=True)           # 32x32 -> 16x16
        self.down5 = UNetBlock(512, 512, down=True)           # 16x16 -> 8x8
        self.down6 = UNetBlock(512, 512, down=True)           # 8x8 -> 4x4
        self.down7 = UNetBlock(512, 512, down=True)           # 4x4 -> 2x2
        
        # Bottleneck - No BatchNorm to avoid 1x1 issues
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),  # 2x2 -> 1x1
            nn.ReLU(inplace=True)
        )
        
        # Decoder (Expanding path) - No BatchNorm in bottleneck upconv
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),  # 1x1 -> 2x2
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)    # 2x2 -> 4x4
        self.up3 = UNetBlock(1024, 512, down=False, use_dropout=True)    # 4x4 -> 8x8
        self.up4 = UNetBlock(1024, 512, down=False)                      # 8x8 -> 16x16
        self.up5 = UNetBlock(1024, 256, down=False)                      # 16x16 -> 32x32
        self.up6 = UNetBlock(512, 128, down=False)                       # 32x32 -> 64x64
        self.up7 = UNetBlock(256, 64, down=False)                        # 64x64 -> 128x128
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),           # 128x128 -> 256x256
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)  # 256 -> 128
        d2 = self.down2(d1)  # 128 -> 64
        d3 = self.down3(d2)  # 64 -> 32
        d4 = self.down4(d3)  # 32 -> 16
        d5 = self.down5(d4)  # 16 -> 8
        d6 = self.down6(d5)  # 8 -> 4
        d7 = self.down7(d6)  # 4 -> 2
        
        # Bottleneck
        bottleneck = self.bottleneck(d7)  # 2 -> 1
        
        # Decoder with skip connections
        u1 = self.up1(bottleneck)  # 1 -> 2
        u2 = self.up2(torch.cat([u1, d7], 1))  # 2 -> 4
        u3 = self.up3(torch.cat([u2, d6], 1))  # 4 -> 8
        u4 = self.up4(torch.cat([u3, d5], 1))  # 8 -> 16
        u5 = self.up5(torch.cat([u4, d4], 1))  # 16 -> 32
        u6 = self.up6(torch.cat([u5, d3], 1))  # 32 -> 64
        u7 = self.up7(torch.cat([u6, d2], 1))  # 64 -> 128
        
        output = self.final(torch.cat([u7, d1], 1))  # 128 -> 256
        
        return output

# PatchGAN Discriminator - FIXED VERSION
class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for Pix2Pix - FIXED VERSION"""
    def __init__(self, input_channels=3, target_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        
        # Input is concatenation of input and target images
        in_channels = input_channels + target_channels
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.model = nn.Sequential(
            discriminator_block(in_channels, 64, normalize=False),    # 256x256 -> 128x128
            discriminator_block(64, 128),                             # 128x128 -> 64x64
            discriminator_block(128, 256),                            # 64x64 -> 32x32
            discriminator_block(256, 512),                            # 32x32 -> 16x16
            nn.Conv2d(512, 1, 4, 1, 1)                               # 16x16 -> 15x15 (PatchGAN output)
        )
    
    def forward(self, input_img, target_img):
        # Concatenate input and target
        x = torch.cat([input_img, target_img], 1)
        return self.model(x)

# Medical Dataset class (keeping the same as original)
class MedicalPix2PixDataset(Dataset):
    """
    Dataset for Pix2Pix training with medical images
    Creates paired data by applying transformations to original images
    """
    def __init__(self, root_dir, transform_input=None, transform_target=None, 
                 augmentation_type='enhancement', class_filter=None):
        self.root_dir = root_dir
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.augmentation_type = augmentation_type
        self.images = []
        self.labels = []
        
        # Collect image paths
        for class_name in os.listdir(root_dir):
            if class_name not in ['benign', 'malignant']:
                continue
            if class_filter and class_name != class_filter:
                continue
                
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(class_name)
    
    def create_input_target_pair(self, image):
        """Create input-target pairs for different medical image tasks"""
        
        if self.augmentation_type == 'enhancement':
            # Input: degraded image, Target: original high-quality image
            input_img = self.degrade_image(image)
            target_img = image.copy()
            
        elif self.augmentation_type == 'denoising':
            # Input: noisy image, Target: clean image
            input_img = self.add_noise(image)
            target_img = image.copy()
            
        elif self.augmentation_type == 'contrast':
            # Input: low contrast, Target: enhanced contrast
            input_img = self.reduce_contrast(image)
            target_img = self.enhance_contrast(image)
            
        elif self.augmentation_type == 'resolution':
            # Input: low resolution, Target: high resolution
            input_img = self.reduce_resolution(image)
            target_img = image.copy()
            
        else:
            # Default: slight variations
            input_img = self.add_slight_variations(image)
            target_img = image.copy()
            
        return input_img, target_img
    
    def degrade_image(self, image):
        """Simulate image degradation"""
        degraded = image.copy()
        
        # Random blur
        if random.random() > 0.3:
            from PIL import ImageFilter
            blur_radius = random.uniform(0.5, 1.5)
            degraded = degraded.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Reduce quality
        if random.random() > 0.4:
            enhancer = ImageEnhance.Sharpness(degraded)
            degraded = enhancer.enhance(random.uniform(0.7, 0.9))
        
        # Add slight noise
        if random.random() > 0.5:
            np_img = np.array(degraded).astype(np.float32)
            noise = np.random.normal(0, 5, np_img.shape)
            np_img = np.clip(np_img + noise, 0, 255)
            degraded = Image.fromarray(np_img.astype(np.uint8))
        
        return degraded
    
    def add_noise(self, image):
        """Add various types of noise"""
        noisy = np.array(image).astype(np.float32)
        
        # Gaussian noise
        if random.random() > 0.3:
            noise_std = random.uniform(8, 20)
            noise = np.random.normal(0, noise_std, noisy.shape)
            noisy += noise
        
        # Salt and pepper noise
        if random.random() > 0.7:
            noise_prob = random.uniform(0.001, 0.01)
            mask = np.random.random(noisy.shape[:2])
            noisy[mask < noise_prob/2] = 0
            noisy[mask > 1 - noise_prob/2] = 255
        
        noisy = np.clip(noisy, 0, 255)
        return Image.fromarray(noisy.astype(np.uint8))
    
    def reduce_contrast(self, image):
        """Reduce image contrast"""
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(0.5, 0.8)
        return enhancer.enhance(factor)
    
    def enhance_contrast(self, image):
        """Enhance image contrast"""
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(1.2, 1.5)
        return enhancer.enhance(factor)
    
    def reduce_resolution(self, image):
        """Simulate low resolution by downsampling and upsampling"""
        original_size = image.size
        # Downsample
        scale_factor = random.uniform(0.25, 0.5)
        small_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
        small_img = image.resize(small_size, Image.LANCZOS)
        # Upsample back
        return small_img.resize(original_size, Image.LANCZOS)
    
    def add_slight_variations(self, image):
        """Add slight variations for general augmentation"""
        varied = image.copy()
        
        # Slight brightness change
        enhancer = ImageEnhance.Brightness(varied)
        varied = enhancer.enhance(random.uniform(0.9, 1.1))
        
        # Slight color change
        enhancer = ImageEnhance.Color(varied)
        varied = enhancer.enhance(random.uniform(0.95, 1.05))
        
        return varied
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Create input-target pair
        input_img, target_img = self.create_input_target_pair(image)
        
        # Apply transforms
        if self.transform_input:
            input_img = self.transform_input(input_img)
        if self.transform_target:
            target_img = self.transform_target(target_img)
        
        return input_img, target_img, self.labels[idx]

# Loss Functions (keeping the same as original)
class Pix2PixLoss(nn.Module):
    """Combined loss for Pix2Pix: GAN loss + L1 loss"""
    def __init__(self, lambda_l1=100):
        super(Pix2PixLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.gan_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
    
    def generator_loss(self, disc_fake, generated, target):
        # GAN loss (fool the discriminator)
        gan_loss = self.gan_loss(disc_fake, torch.ones_like(disc_fake))
        # L1 reconstruction loss
        l1_loss = self.l1_loss(generated, target)
        # Combined loss
        total_loss = gan_loss + self.lambda_l1 * l1_loss
        return total_loss, gan_loss, l1_loss
    
    def discriminator_loss(self, disc_real, disc_fake):
        # Real loss
        real_loss = self.gan_loss(disc_real, torch.ones_like(disc_real))
        # Fake loss
        fake_loss = self.gan_loss(disc_fake, torch.zeros_like(disc_fake))
        # Combined loss
        total_loss = (real_loss + fake_loss) * 0.5
        return total_loss

# Updated training function with proper error handling
def train_pix2pix(train_directory, epochs=EPOCHS, save_interval=25, 
                  augmentation_type='enhancement', class_specific=True):
    """
    Train Pix2Pix for medical image translation - FIXED VERSION
    """
    
    # Transforms
    transform_input = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    transform_target = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    models = {}
    classes_to_train = ['benign', 'malignant'] if class_specific else [None]
    
    for class_name in classes_to_train:
        print(f"\n{'='*60}")
        if class_name:
            print(f"Training Pix2Pix for {class_name} class ({augmentation_type})")
        else:
            print(f"Training Pix2Pix for all classes ({augmentation_type})")
        print(f"{'='*60}")
        
        # Create dataset
        dataset = MedicalPix2PixDataset(
            root_dir=train_directory,
            transform_input=transform_input,
            transform_target=transform_target,
            augmentation_type=augmentation_type,
            class_filter=class_name
        )
        
        if len(dataset) == 0:
            print(f"No images found for {class_name or 'all classes'}")
            continue
        
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                               num_workers=2, pin_memory=True)
        
        print(f"Dataset size: {len(dataset)} images")
        print(f"Task: {augmentation_type}")
        
        # Initialize networks
        generator = Pix2PixGenerator(CHANNELS, CHANNELS).to(device)
        discriminator = PatchGANDiscriminator(CHANNELS, CHANNELS).to(device)
        
        # Test forward pass with proper batch size
        try:
            # Use a batch size > 1 for testing to avoid BatchNorm issues
            test_batch_size = min(4, BATCH_SIZE)
            test_input = torch.randn(test_batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            test_target = torch.randn(test_batch_size, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            
            # Set to eval mode to avoid BatchNorm training issues during test
            generator.eval()
            discriminator.eval()
            
            with torch.no_grad():
                test_output = generator(test_input)
                test_disc = discriminator(test_input, test_target)
            
            # Set back to train mode
            generator.train()
            discriminator.train()
            
            print(f"Network test successful:")
            print(f"  Generator output: {test_output.shape}")
            print(f"  Discriminator output: {test_disc.shape}")
        except Exception as e:
            print(f"Network test failed: {e}")
            continue
        
        # Optimizers
        opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
        opt_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))
        
        # Schedulers
        scheduler_gen = optim.lr_scheduler.ExponentialLR(opt_gen, gamma=0.995)
        scheduler_disc = optim.lr_scheduler.ExponentialLR(opt_disc, gamma=0.995)
        
        # Loss function
        criterion = Pix2PixLoss(lambda_l1=LAMBDA_L1)
        
        # Training metrics
        gen_losses = []
        disc_losses = []
        l1_losses = []
        
        # Fixed samples for monitoring
        fixed_samples = None
        for batch_idx, (input_imgs, target_imgs, _) in enumerate(dataloader):
            if batch_idx == 0:
                fixed_samples = (input_imgs[:4].to(device), target_imgs[:4].to(device))
                break
        
        # Create directories
        class_suffix = f"_{class_name}_{augmentation_type}" if class_name else f"_all_{augmentation_type}"
        os.makedirs(f"pix2pix_models{class_suffix}", exist_ok=True)
        os.makedirs(f"pix2pix_samples{class_suffix}", exist_ok=True)
        
        print(f"Starting Pix2Pix training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            epoch_l1_loss = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (input_imgs, target_imgs, _) in enumerate(progress_bar):
                input_imgs = input_imgs.to(device)
                target_imgs = target_imgs.to(device)
                batch_size_current = input_imgs.size(0)
                
                # Skip batches with size 1 to avoid BatchNorm issues
                if batch_size_current == 1:
                    continue
                
                # Train Generator
                generator.zero_grad()
                
                # Generate fake images
                generated_imgs = generator(input_imgs)
                
                # Discriminator's opinion on generated images
                disc_fake = discriminator(input_imgs, generated_imgs)
                
                # Generator loss
                gen_loss_total, gen_gan_loss, gen_l1_loss = criterion.generator_loss(
                    disc_fake, generated_imgs, target_imgs
                )
                
                gen_loss_total.backward()
                opt_gen.step()
                
                # Train Discriminator
                discriminator.zero_grad()
                
                # Real images
                disc_real = discriminator(input_imgs, target_imgs)
                
                # Fake images (detached to avoid backprop through generator)
                disc_fake = discriminator(input_imgs, generated_imgs.detach())
                
                # Discriminator loss
                disc_loss = criterion.discriminator_loss(disc_real, disc_fake)
                
                disc_loss.backward()
                opt_disc.step()
                
                # Update metrics
                epoch_gen_loss += gen_loss_total.item()
                epoch_disc_loss += disc_loss.item()
                epoch_l1_loss += gen_l1_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'G_Loss': f'{gen_loss_total.item():.3f}',
                    'D_Loss': f'{disc_loss.item():.3f}',
                    'L1': f'{gen_l1_loss.item():.3f}'
                })
            
            # Calculate averages
            avg_gen_loss = epoch_gen_loss / max(len(dataloader), 1)
            avg_disc_loss = epoch_disc_loss / max(len(dataloader), 1)
            avg_l1_loss = epoch_l1_loss / max(len(dataloader), 1)
            
            gen_losses.append(avg_gen_loss)
            disc_losses.append(avg_disc_loss)
            l1_losses.append(avg_l1_loss)
            
            # Learning rate scheduling
            if epoch > 100:
                scheduler_gen.step()
                scheduler_disc.step()
            
            # Save samples
            if (epoch + 1) % save_interval == 0:
                generator.eval()
                with torch.no_grad():
                    if fixed_samples is not None:
                        fixed_input, fixed_target = fixed_samples
                        generated_samples = generator(fixed_input)
                        
                        # Denormalize for visualization
                        input_display = (fixed_input + 1) / 2
                        target_display = (fixed_target + 1) / 2
                        generated_display = (generated_samples + 1) / 2
                        
                        # Create comparison grid
                        comparison = torch.cat([
                            input_display, 
                            generated_display, 
                            target_display
                        ], dim=0)
                        
                        vutils.save_image(comparison, 
                                        f"pix2pix_samples{class_suffix}/epoch_{epoch+1}.png", 
                                        nrow=4, normalize=False, padding=2)
                generator.train()
            
            # Progress report
            if (epoch + 1) % 25 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Generator Loss: {avg_gen_loss:.4f}")
                print(f"Discriminator Loss: {avg_disc_loss:.4f}")
                print(f"L1 Loss: {avg_l1_loss:.4f}")
        
        # Save final models
        torch.save(generator.state_dict(), f"pix2pix_models{class_suffix}/generator.pth")
        torch.save(discriminator.state_dict(), f"pix2pix_models{class_suffix}/discriminator.pth")
        
        # Plot training curves
        if len(gen_losses) > 0:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(gen_losses, label='Generator Total', alpha=0.7)
            plt.plot(disc_losses, label='Discriminator', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Pix2Pix Losses ({class_name or "all"})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 2)
            plt.plot(l1_losses, label='L1 Reconstruction', color='green', alpha=0.7)
            plt.xlabel('Epoch')
            plt.ylabel('L1 Loss')
            plt.title('Reconstruction Quality')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 3, 3)
            # Loss balance
            if len(gen_losses) > 5:
                balance = np.array(gen_losses) / (np.array(disc_losses) + 1e-8)
                plt.plot(balance, label='G/D Ratio', color='purple')
                plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
                plt.xlabel('Epoch')
                plt.ylabel('Loss Ratio')
                plt.title('Training Balance')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'pix2pix_curves{class_suffix}.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        models[class_name or 'all'] = {
            'generator': generator,
            'discriminator': discriminator,
            'class_suffix': class_suffix
        }
        
        print(f"Pix2Pix training completed for {class_name or 'all classes'}")
    
        return models

# Generate dataset using trained Pix2Pix models
def generate_pix2pix_dataset(models, train_directory, output_directory, 
                            augmentation_type='enhancement', images_per_original=8):
    """Generate augmented dataset using Pix2Pix translation"""
    
    print("=== Generating Pix2Pix Augmented Dataset ===")
    print(f"Available models: {list(models.keys())}")
    
    # Create output directories
    for class_name in ['benign', 'malignant']:
        os.makedirs(os.path.join(output_directory, class_name), exist_ok=True)
    
    total_generated = 0
    generation_stats = {'benign': 0, 'malignant': 0}
    
    # Setup transforms for input generation
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
        
        # Get appropriate model - try multiple strategies
        generator = None
        
        # Strategy 1: Try class-specific model
        if class_name in models:
            generator = models[class_name]['generator']
            print(f"Using class-specific model for {class_name}")
        # Strategy 2: Try 'all' model
        elif 'all' in models:
            generator = models['all']['generator']
            print(f"Using 'all' model for {class_name}")
        # Strategy 3: Use any available model
        elif len(models) > 0:
            first_model_key = list(models.keys())[0]
            generator = models[first_model_key]['generator']
            print(f"Using available model '{first_model_key}' for {class_name}")
        
        if generator is None:
            print(f"ERROR: No model available for {class_name} class")
            continue
        
        generator.eval()
        
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
                print(f"Saved original: {orig_output}")
                
                with torch.no_grad():
                    # Generate multiple variations
                    for i in range(images_per_original):
                        try:
                            # Create different input conditions
                            dataset_temp = MedicalPix2PixDataset(
                                train_directory, 
                                augmentation_type=augmentation_type
                            )
                            
                            # Generate degraded input
                            if augmentation_type == 'enhancement':
                                input_img = dataset_temp.degrade_image(original_img)
                            elif augmentation_type == 'denoising':
                                input_img = dataset_temp.add_noise(original_img)
                            elif augmentation_type == 'contrast':
                                input_img = dataset_temp.reduce_contrast(original_img)
                            elif augmentation_type == 'resolution':
                                input_img = dataset_temp.reduce_resolution(original_img)
                            else:
                                input_img = dataset_temp.add_slight_variations(original_img)
                            
                            # Transform for model
                            input_tensor = transform_for_generation(input_img).unsqueeze(0).to(device)
                            
                            # Generate enhanced/translated image
                            generated_tensor = generator(input_tensor)
                            
                            # Convert back to PIL
                            generated_tensor = (generated_tensor + 1) / 2  # Denormalize
                            generated_tensor = torch.clamp(generated_tensor, 0, 1)
                            
                            generated_img = transforms.ToPILImage()(generated_tensor.squeeze(0).cpu())
                            
                            # Save generated image
                            gen_name = f"pix2pix_{augmentation_type}_{base_name}_{i+1:02d}.png"
                            gen_path = os.path.join(output_directory, class_name, gen_name)
                            generated_img.save(gen_path)
                            
                            total_generated += 1
                            generation_stats[class_name] += 1
                            successful_generations += 1
                            
                            # Save the input (degraded) image for comparison (only once per original)
                            if i == 0:
                                input_name = f"input_{augmentation_type}_{base_name}.png"
                                input_path = os.path.join(output_directory, class_name, input_name)
                                input_img.save(input_path)
                        
                        except Exception as inner_e:
                            print(f"Failed to generate variation {i+1} for {img_name}: {str(inner_e)}")
                            failed_generations += 1
                            continue
                
                # Progress report for every 10 images
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
        
        # Set back to train mode
        generator.train()
    
    # Print statistics
    print(f"\n=== Pix2Pix Dataset Generation Complete ===")
    print(f"Total generated images: {total_generated}")
    for class_name, count in generation_stats.items():
        if os.path.exists(os.path.join(train_directory, class_name)):
            original_count = len([f for f in os.listdir(os.path.join(train_directory, class_name))
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            print(f"{class_name}: {original_count} original + {count} generated = {original_count + count} total")
            print(f"  Augmentation factor: {count/original_count:.1f}x")
    
    return generation_stats

# Evaluation and utility functions
def evaluate_pix2pix_quality(models, test_samples=50):
    """Evaluate Pix2Pix model quality using various metrics"""
    
    print("\n=== Pix2Pix Quality Evaluation ===")
    
    for model_name, model_info in models.items():
        generator = model_info['generator']
        generator.eval()
        
        print(f"\nEvaluating {model_name} model...")
        
        # Generate test samples
        with torch.no_grad():
            # Create random inputs
            test_inputs = torch.randn(test_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            generated_outputs = generator(test_inputs)
            
            # Calculate metrics
            mse_scores = []
            for i in range(test_samples):
                input_img = test_inputs[i].cpu()
                output_img = generated_outputs[i].cpu()
                
                # MSE between input and output
                mse = F.mse_loss(input_img, output_img).item()
                mse_scores.append(mse)
            
            avg_mse = np.mean(mse_scores)
            std_mse = np.std(mse_scores)
            
            print(f"  Average MSE: {avg_mse:.4f} ± {std_mse:.4f}")
            
            # Check for mode collapse
            pairwise_distances = []
            sample_size = min(20, test_samples)
            for i in range(sample_size):
                for j in range(i+1, sample_size):
                    dist = F.mse_loss(generated_outputs[i], generated_outputs[j]).item()
                    pairwise_distances.append(dist)
            
            avg_diversity = np.mean(pairwise_distances)
            print(f"  Output diversity: {avg_diversity:.4f}")
            
            diversity_status = "High" if avg_diversity > 0.1 else "Medium" if avg_diversity > 0.05 else "Low"
            print(f"  Diversity assessment: {diversity_status}")
        
        generator.train()

def create_pix2pix_demo(models, output_path="pix2pix_demonstrations"):
    """Create demonstration images showing Pix2Pix translation capabilities"""
    
    print("\n=== Creating Pix2Pix Demonstration ===")
    os.makedirs(output_path, exist_ok=True)
    
    for model_name, model_info in models.items():
        generator = model_info['generator']
        generator.eval()
        
        with torch.no_grad():
            # Create various input conditions
            demo_inputs = []
            
            # Random noise input
            noise_input = torch.randn(4, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            demo_inputs.append(("Random_Noise", noise_input))
            
            # Structured patterns
            pattern_input = torch.zeros(4, CHANNELS, IMAGE_SIZE, IMAGE_SIZE).to(device)
            for i in range(4):
                # Create different patterns for each sample
                pattern_input[i, :, 50:100, 50:100] = 1.0  # White square
                pattern_input[i, :, 150:200, 150:200] = -1.0  # Black square
            demo_inputs.append(("Pattern_Input", pattern_input))
            
            # Generate translations
            for demo_name, input_tensor in demo_inputs:
                output_tensor = generator(input_tensor)
                
                # Denormalize for display
                input_display = (input_tensor + 1) / 2
                output_display = (output_tensor + 1) / 2
                
                # Create side-by-side comparison
                comparison = torch.cat([input_display, output_display], dim=3)  # Concatenate horizontally
                
                vutils.save_image(comparison, 
                                f"{output_path}/{model_name}_{demo_name}.png",
                                nrow=2, normalize=False, padding=2)
        
        generator.train()
    
    print(f"Demo images saved to {output_path}/")

def validate_dataset(directory):
    """Validate dataset structure and count images"""
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

def get_user_configuration():
    """Get configuration from user or use defaults"""
    config = {
        'train_directory': r"F:\jihad\Breast GAN types\breast\breast\split_dataset\train",
        'output_directory': r"F:\jihad\Breast GAN types\breast\breast\pix2pix_augmented",
        'augmentation_type': 'enhancement',
        'epochs': 1000,  # Reduced for faster testing
        'class_specific': True,
        'images_per_original': 10,  # Generate 10 images per original image
        'save_interval': 10
    }
    
    return config

def main():
    """Main function for Pix2Pix training and evaluation"""
    
    print("=" * 80)
    print("PIX2PIX GAN FOR MEDICAL IMAGE TRANSLATION")
    print("Conditional image-to-image translation for medical imaging")
    print("=" * 80)
    
    # Get configuration
    config = get_user_configuration()
    
    print(f"\n=== Configuration ===")
    print(f"Device: {device}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {config['epochs']}")
    print(f"Augmentation type: {config['augmentation_type']}")
    print(f"Training directory: {config['train_directory']}")
    print(f"Output directory: {config['output_directory']}")
    
    # Validate dataset
    print(f"\n=== Dataset Validation ===")
    is_valid, class_info = validate_dataset(config['train_directory'])
    
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
    
    if total_images < 10:
        print("Warning: Very few images found. Training may not be effective.")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            return False
    
    # Show available augmentation types
    print(f"\n=== Augmentation Types ===")
    augmentation_info = {
        'enhancement': 'Improve degraded/blurry images to high quality',
        'denoising': 'Remove noise from corrupted images',
        'contrast': 'Enhance contrast from low-contrast inputs',
        'resolution': 'Super-resolution from low-resolution inputs'
    }
    
    for aug_type, description in augmentation_info.items():
        marker = " [SELECTED]" if aug_type == config['augmentation_type'] else ""
        print(f"  {aug_type}: {description}{marker}")
    
    # Training phase
    print(f"\n=== Training Phase ===")
    print("Starting Pix2Pix training...")
    
    try:
        trained_models = train_pix2pix(
            train_directory=config['train_directory'],
            epochs=config['epochs'],
            save_interval=config['save_interval'],
            augmentation_type=config['augmentation_type'],
            class_specific=config['class_specific']
        )
        
        if not trained_models:
            print("ERROR: No models were successfully trained")
            print("Common issues:")
            print("- Insufficient GPU memory (try reducing BATCH_SIZE)")
            print("- Dataset path issues")
            print("- Image format compatibility")
            return False
        
        print(f"SUCCESS: Training completed for {len(trained_models)} model(s)")
        
    except Exception as e:
        print(f"TRAINING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Quality evaluation
    print(f"\n=== Quality Evaluation ===")
    try:
        evaluate_pix2pix_quality(trained_models, test_samples=20)
    except Exception as e:
        print(f"Quality evaluation failed: {str(e)}")
    
    # Create demonstrations
    print(f"\n=== Creating Demonstrations ===")
    try:
        create_pix2pix_demo(trained_models, output_path="pix2pix_demonstrations")
        print("Demonstration images created successfully")
    except Exception as e:
        print(f"Demo creation failed: {str(e)}")
    
    # Generate augmented dataset
    print(f"\n=== Dataset Generation ===")
    try:
        generation_stats = generate_pix2pix_dataset(
            models=trained_models,
            train_directory=config['train_directory'],
            output_directory=config['output_directory'],
            augmentation_type=config['augmentation_type'],
            images_per_original=config['images_per_original']
        )
        
        print("Augmented dataset generated successfully")
        
    except Exception as e:
        print(f"Dataset generation failed: {str(e)}")
        return False
    
    # Results summary
    print(f"\n=== Training Results Summary ===")
    print("Generated files and directories:")
    
    file_info = [
        ("pix2pix_models_*/", "Trained generator and discriminator models"),
        ("pix2pix_samples_*/", "Training progress images"),
        ("pix2pix_curves_*.png", "Training loss curves and metrics"),
        ("pix2pix_demonstrations/", "Model capability demonstrations"),
        (config['output_directory'], "Augmented dataset for classification")
    ]
    
    for filename, description in file_info:
        print(f"  {filename:<30} - {description}")
    
    print(f"\n=== Pix2Pix Advantages for Medical Imaging ===")
    advantages = [
        "Conditional generation preserves anatomical structure",
        "L1 loss ensures pixel-level reconstruction accuracy",
        "PatchGAN discriminator maintains local texture quality",
        "Task-specific capabilities (enhancement, denoising, etc.)",
        "More stable training than unconditional GANs",
        "Controllable image-to-image translation"
    ]
    
    for advantage in advantages:
        print(f"  • {advantage}")
    
    print(f"\n=== Next Steps ===")
    next_steps = [
        "Review training samples in pix2pix_samples folders",
        "Examine demonstration images for quality assessment",
        "Use augmented dataset to train classification models",
        "Experiment with different augmentation types",
        "Compare results with original dataset performance",
        "Consider fine-tuning for specific medical imaging tasks"
    ]
    
    for i, step in enumerate(next_steps, 1):
        print(f"  {i}. {step}")
    
    print(f"\n=== Expected Performance Impact ===")
    print("The Pix2Pix augmented dataset should provide:")
    print(f"  • {config['images_per_original']}x more training data per original image")
    print("  • Better generalization through realistic variations")
    print("  • Improved robustness to image quality variations")
    print("  • Enhanced classification performance on medical images")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 80)
        print("PIX2PIX PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nYour Pix2Pix models are ready for medical image translation!")
        print("Use the augmented dataset to improve classification performance!")
    else:
        print("\n" + "=" * 80)
        print("PIX2PIX PIPELINE ENCOUNTERED ERRORS")
        print("=" * 80)
        print("\nPlease check the error messages above and:")
        print("1. Verify dataset paths and structure")
        print("2. Ensure sufficient GPU memory")
        print("3. Check image file formats and accessibility")
        print("4. Consider reducing batch size or image resolution")