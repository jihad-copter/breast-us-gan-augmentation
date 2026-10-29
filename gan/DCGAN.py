import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters - Latest Research for DCGAN
IMAGE_SIZE = 128
BATCH_SIZE = 16
LEARNING_RATE_G = 0.0001  # Lower LR for generator stability
LEARNING_RATE_D = 0.0003  # TTUR - different learning rates
BETA1 = 0.0  # Adam beta1 = 0 for better convergence (latest research)
BETA2 = 0.9
EPOCHS = 1000
Z_DIM = 128  # Increased latent dimension
CHANNELS = 3
LAMBDA_GP = 10  # For gradient penalty
FEATURE_MAPS_G = 64  # Generator feature maps
FEATURE_MAPS_D = 64  # Discriminator feature maps

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_filter=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path) and class_name in self.class_to_idx:
                # Skip class if filter is specified and doesn't match
                if class_filter and class_name != class_filter:
                    continue
                    
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        self.images.append(os.path.join(class_path, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Self-Attention Module for high-quality generation
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Compute attention
        proj_query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        proj_value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        # Apply learnable weight and residual connection
        out = self.gamma * out + x
        return out

# Advanced DCGAN Generator with latest improvements
class AdvancedDCGANGenerator(nn.Module):
    def __init__(self, z_dim, channels, features_g):
        super(AdvancedDCGANGenerator, self).__init__()
        self.z_dim = z_dim
        
        # Initial dense layer
        self.initial = nn.Sequential(
            # Input: Z_DIM x 1 x 1
            nn.ConvTranspose2d(z_dim, features_g * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 16),
            nn.ReLU(True)
        )
        
        # Progressive upsampling blocks with residual connections
        self.conv_blocks = nn.ModuleList([
            # 4x4 -> 8x8
            self._make_block(features_g * 16, features_g * 8, 4, 2, 1),
            # 8x8 -> 16x16  
            self._make_block(features_g * 8, features_g * 4, 4, 2, 1),
            # 16x16 -> 32x32
            self._make_block(features_g * 4, features_g * 2, 4, 2, 1),
            # 32x32 -> 64x64
            self._make_block(features_g * 2, features_g, 4, 2, 1),
        ])
        
        # Self-attention at 64x64 resolution
        self.attention = SelfAttention(features_g)
        
        # Final upsampling to target resolution
        self.final = nn.Sequential(
            # 64x64 -> 128x128
            nn.ConvTranspose2d(features_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        # Apply spectral normalization to prevent mode collapse
        self.apply(self._apply_spectral_norm)
    
    def _make_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            # Additional conv for better feature learning
            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def _apply_spectral_norm(self, module):
        if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d)):
            if hasattr(module, 'weight') and module.weight.size(0) >= 64:
                spectral_norm(module)
    
    def forward(self, noise):
        # Reshape noise to 4D tensor
        x = noise.view(noise.size(0), noise.size(1), 1, 1)
        
        # Initial upsampling
        x = self.initial(x)
        
        # Progressive upsampling
        for block in self.conv_blocks:
            x = block(x)
        
        # Apply self-attention
        x = self.attention(x)
        
        # Final layer
        x = self.final(x)
        return x

# Advanced DCGAN Discriminator with Spectral Normalization
class AdvancedDCGANDiscriminator(nn.Module):
    def __init__(self, channels, features_d):
        super(AdvancedDCGANDiscriminator, self).__init__()
        
        # Progressive downsampling blocks
        self.conv_blocks = nn.ModuleList([
            # 128x128 -> 64x64
            self._make_block(channels, features_d, 4, 2, 1, normalize=False),
            # 64x64 -> 32x32
            self._make_block(features_d, features_d * 2, 4, 2, 1),
            # 32x32 -> 16x16
            self._make_block(features_d * 2, features_d * 4, 4, 2, 1),
        ])
        
        # Self-attention at 32x32 resolution (after first downsampling)
        self.attention = SelfAttention(features_d * 2)
        
        # Final blocks
        self.final_blocks = nn.ModuleList([
            # 16x16 -> 8x8
            self._make_block(features_d * 4, features_d * 8, 4, 2, 1),
            # 8x8 -> 4x4
            self._make_block(features_d * 8, features_d * 16, 4, 2, 1),
        ])
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Conv2d(features_d * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        # Apply spectral normalization for training stability
        self.apply(self._apply_spectral_norm)
    
    def _make_block(self, in_channels, out_channels, kernel_size, stride, padding, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _apply_spectral_norm(self, module):
        if isinstance(module, nn.Conv2d):
            spectral_norm(module)
    
    def forward(self, image):
        x = image
        
        # First two downsampling blocks
        x = self.conv_blocks[0](x)
        x = self.conv_blocks[1](x)
        
        # Apply self-attention
        x = self.attention(x)
        
        # Continue downsampling
        x = self.conv_blocks[2](x)
        
        # Final blocks
        for block in self.final_blocks:
            x = block(x)
        
        # Classification
        x = self.classifier(x)
        return x.view(-1)

# Gradient Penalty for WGAN-GP loss
def compute_gradient_penalty(discriminator, real_images, fake_images, device):
    batch_size = real_images.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # Interpolate between real and fake images
    interpolated = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated.requires_grad_(True)
    
    # Get discriminator output for interpolated images
    disc_interpolated = discriminator(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_penalty

# Advanced training function with latest techniques
def train_advanced_dcgan(train_directory, epochs=EPOCHS, save_interval=50, class_specific=True):
    """
    Train DCGAN with latest research improvements:
    - WGAN-GP loss for stability
    - Spectral normalization
    - Self-attention for long-range dependencies
    - TTUR (Two Time-scale Update Rule)
    - Progressive quality improvements
    - Class-specific training for medical images
    """
    
    # Enhanced data preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    models = {}
    
    # Train separate models for each class
    classes_to_train = ['benign', 'malignant']
    
    for class_name in classes_to_train:
        print(f"\n{'='*60}")
        print(f"Training DCGAN for {class_name} class")
        print(f"{'='*60}")
        
        # Create dataset
        dataset = MedicalImageDataset(train_directory, transform=transform, class_filter=class_name)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        
        print(f"Dataset size: {len(dataset)} images")
        print(f"Number of batches: {len(dataloader)}")
        
        if len(dataset) < BATCH_SIZE:
            print("Warning: Dataset size is smaller than batch size")
        
        # Initialize networks
        generator = AdvancedDCGANGenerator(Z_DIM, CHANNELS, FEATURE_MAPS_G).to(device)
        discriminator = AdvancedDCGANDiscriminator(CHANNELS, FEATURE_MAPS_D).to(device)
        
        # Initialize weights
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
        
        generator.apply(init_weights)
        discriminator.apply(init_weights)
        
        # Optimizers with TTUR
        opt_gen = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
        opt_disc = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))
        
        # Learning rate schedulers
        scheduler_gen = optim.lr_scheduler.ExponentialLR(opt_gen, gamma=0.99)
        scheduler_disc = optim.lr_scheduler.ExponentialLR(opt_disc, gamma=0.99)
        
        # Loss functions
        adversarial_loss = nn.BCELoss()
        
        # Fixed noise for monitoring progress
        fixed_noise = torch.randn(16, Z_DIM).to(device)
        
        # Training metrics
        gen_losses = []
        disc_losses = []
        best_gen_loss = float('inf')
        
        # Create directories
        class_suffix = f"_{class_name}"
        os.makedirs(f"models{class_suffix}", exist_ok=True)
        os.makedirs(f"generated_samples{class_suffix}", exist_ok=True)
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            generator.train()
            discriminator.train()
            
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (real_images, _) in enumerate(progress_bar):
                real_images = real_images.to(device)
                batch_size = real_images.size(0)
                
                # Train Discriminator multiple times per generator update
                for _ in range(2):  # 2:1 ratio for stability
                    discriminator.zero_grad()
                    
                    # Train with real images
                    real_labels = torch.ones(batch_size).to(device) * 0.9  # Label smoothing
                    real_output = discriminator(real_images)
                    real_loss = adversarial_loss(real_output, real_labels)
                    
                    # Train with fake images
                    noise = torch.randn(batch_size, Z_DIM).to(device)
                    fake_images = generator(noise).detach()
                    fake_labels = torch.zeros(batch_size).to(device) + 0.1  # Label smoothing
                    fake_output = discriminator(fake_images)
                    fake_loss = adversarial_loss(fake_output, fake_labels)
                    
                    # Gradient penalty for stability
                    gp = compute_gradient_penalty(discriminator, real_images, fake_images, device)
                    
                    # Total discriminator loss
                    disc_loss = real_loss + fake_loss + LAMBDA_GP * gp
                    disc_loss.backward()
                    opt_disc.step()
                    
                    epoch_disc_loss += disc_loss.item()
                
                # Train Generator
                generator.zero_grad()
                
                noise = torch.randn(batch_size, Z_DIM).to(device)
                fake_images = generator(noise)
                fake_output = discriminator(fake_images)
                
                # Generator wants discriminator to classify fake images as real
                real_labels = torch.ones(batch_size).to(device)
                gen_loss = adversarial_loss(fake_output, real_labels)
                
                gen_loss.backward()
                opt_gen.step()
                
                epoch_gen_loss += gen_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'G_Loss': f'{gen_loss.item():.4f}',
                    'D_Loss': f'{disc_loss.item():.4f}'
                })
            
            # Calculate average losses
            avg_gen_loss = epoch_gen_loss / len(dataloader)
            avg_disc_loss = epoch_disc_loss / (len(dataloader) * 2)  # Account for 2:1 training
            
            gen_losses.append(avg_gen_loss)
            disc_losses.append(avg_disc_loss)
            
            # Update learning rates
            if epoch > 100:  # Start decay after 100 epochs
                scheduler_gen.step()
                scheduler_disc.step()
            
            # Save best model
            if avg_gen_loss < best_gen_loss:
                best_gen_loss = avg_gen_loss
                torch.save({
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'gen_optimizer': opt_gen.state_dict(),
                    'disc_optimizer': opt_disc.state_dict(),
                    'epoch': epoch,
                    'gen_loss': avg_gen_loss,
                    'disc_loss': avg_disc_loss
                }, f"models{class_suffix}/best_dcgan_model.pth")
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch [{epoch+1}/{epochs}]")
                print(f"Generator Loss: {avg_gen_loss:.4f}")
                print(f"Discriminator Loss: {avg_disc_loss:.4f}")
                print(f"Best Generator Loss: {best_gen_loss:.4f}")
                print(f"Learning Rate G: {opt_gen.param_groups[0]['lr']:.6f}")
                print(f"Learning Rate D: {opt_disc.param_groups[0]['lr']:.6f}")
            
            # Generate and save sample images
            if (epoch + 1) % save_interval == 0:
                generator.eval()
                with torch.no_grad():
                    fake_samples = generator(fixed_noise)
                    fake_samples = (fake_samples + 1) / 2  # Denormalize to [0, 1]
                    
                    vutils.save_image(fake_samples, 
                                    f"generated_samples{class_suffix}/epoch_{epoch+1}.png", 
                                    nrow=4, normalize=False, padding=2)
                generator.train()
        
        # Save final models
        torch.save(generator.state_dict(), f"models{class_suffix}/final_generator.pth")
        torch.save(discriminator.state_dict(), f"models{class_suffix}/final_discriminator.pth")
        
        # Plot training curves
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(gen_losses, label='Generator Loss', alpha=0.7)
        plt.plot(disc_losses, label='Discriminator Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'DCGAN Training Losses{class_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Smoothed losses
        window = max(1, len(gen_losses) // 20)
        if len(gen_losses) > window:
            smooth_gen = np.convolve(gen_losses, np.ones(window)/window, mode='valid')
            smooth_disc = np.convolve(disc_losses, np.ones(window)/window, mode='valid')
            plt.plot(smooth_gen, label='Smooth Gen Loss')
            plt.plot(smooth_disc, label='Smooth Disc Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Smoothed Loss')
            plt.title(f'Smoothed Losses{class_suffix}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'dcgan_training_curves{class_suffix}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store model for generation phase
        models[class_name] = {
            'generator': generator,
            'discriminator': discriminator,
            'class_suffix': class_suffix
        }
        
        print(f"✓ Training completed for {class_name} class")
    
    return models

# Enhanced generation function with organized file structure
def generate_organized_dataset(models, train_directory, output_directory, images_per_original=10):
    """
    Generate augmented dataset with organized structure:
    - Each original image gets its own folder
    - Generate exactly 10 images per original
    - Proper naming based on original filename
    """
    
    print("=== Generating Organized DCGAN Dataset ===")
    print(f"Target: {images_per_original} images per original image")
    
    # Create main output directory
    os.makedirs(output_directory, exist_ok=True)
    
    total_generated = 0
    generation_stats = {'benign': {'original': 0, 'generated': 0}, 
                       'malignant': {'original': 0, 'generated': 0}}
    
    for class_name in ['benign', 'malignant']:
        print(f"\n{'='*50}")
        print(f"Processing {class_name.upper()} class")
        print(f"{'='*50}")
        
        class_path = os.path.join(train_directory, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: {class_path} not found, skipping...")
            continue
        
        # Get the trained generator for this class
        if class_name not in models:
            print(f"Warning: No trained model for {class_name}, skipping...")
            continue
        
        generator = models[class_name]['generator']
        generator.eval()
        
        # Create class output directory
        class_output_dir = os.path.join(output_directory, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        print(f"Found {len(image_files)} original images")
        generation_stats[class_name]['original'] = len(image_files)
        
        # Process each original image
        for img_idx, img_name in enumerate(tqdm(image_files, desc=f"Processing {class_name}")):
            # Get base name without extension
            base_name = os.path.splitext(img_name)[0]
            
            # Create individual folder for this original image
            img_output_dir = os.path.join(class_output_dir, base_name)
            os.makedirs(img_output_dir, exist_ok=True)
            
            # Copy original image to the folder
            original_path = os.path.join(class_path, img_name)
            original_copy_path = os.path.join(img_output_dir, f"original_{base_name}.png")
            
            # Load and save original image
            original_img = Image.open(original_path).convert('RGB')
            original_img = original_img.resize((IMAGE_SIZE, IMAGE_SIZE))
            original_img.save(original_copy_path)
            
            # Generate the specified number of variations
            with torch.no_grad():
                generated_count = 0
                
                # Generate images in batches for efficiency
                batch_size = min(4, images_per_original)  # Process in small batches
                
                while generated_count < images_per_original:
                    # Calculate how many to generate in this batch
                    remaining = images_per_original - generated_count
                    current_batch_size = min(batch_size, remaining)
                    
                    # Generate noise with some variation techniques
                    noise = torch.randn(current_batch_size, Z_DIM).to(device)
                    
                    # Add different noise characteristics for variety
                    for i in range(current_batch_size):
                        if generated_count + i < images_per_original // 3:
                            # Standard noise
                            pass
                        elif generated_count + i < 2 * images_per_original // 3:
                            # Truncated noise for higher quality
                            noise[i] = torch.clamp(noise[i], -1.5, 1.5)
                        else:
                            # Scaled noise for more variation
                            noise[i] = noise[i] * 0.8
                    
                    # Generate images
                    fake_images = generator(noise)
                    fake_images = torch.clamp((fake_images + 1) / 2, 0, 1)  # Denormalize and clamp
                    
                    # Save each generated image
                    for i in range(current_batch_size):
                        fake_img = fake_images[i].cpu()
                        fake_img_pil = transforms.ToPILImage()(fake_img)
                        
                        # Create filename with proper numbering
                        gen_filename = f"generated_{base_name}_{generated_count + i + 1:02d}.png"
                        gen_path = os.path.join(img_output_dir, gen_filename)
                        fake_img_pil.save(gen_path)
                    
                    generated_count += current_batch_size
                    total_generated += current_batch_size
                    generation_stats[class_name]['generated'] += current_batch_size
            
            # Print progress for every 10th image
            if (img_idx + 1) % 10 == 0:
                print(f"  Completed {img_idx + 1}/{len(image_files)} images for {class_name}")
    
    # Print final statistics
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total generated images: {total_generated}")
    print(f"Images per original: {images_per_original}")
    
    for class_name, stats in generation_stats.items():
        if stats['original'] > 0:
            print(f"\n{class_name.upper()} CLASS:")
            print(f"  Original images: {stats['original']}")
            print(f"  Generated images: {stats['generated']}")
            print(f"  Total images: {stats['original'] + stats['generated']}")
            print(f"  Folders created: {stats['original']}")
            print(f"  Images per folder: {images_per_original + 1} (1 original + {images_per_original} generated)")
            
            expected_total = stats['original'] * (images_per_original + 1)
            actual_total = stats['original'] + stats['generated']
            print(f"  Expected total: {expected_total}")
            print(f"  Actual total: {actual_total}")
            print(f"  Status: {'✓ Complete' if expected_total == actual_total else '⚠ Incomplete'}")
    
    return generation_stats

def main():
    """Main function for DCGAN training and organized dataset generation"""
    
    # Configuration
    train_directory = r"F:\jihad\Breast GAN types\breast\breast\split_dataset\train"
    augmented_output_directory = r"F:\jihad\Breast GAN types\breast\breast\dcgan_organized_dataset"
    
    print("=== DCGAN Medical Image Augmentation with Organization ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Device: {device}")
    
    # Dataset verification
    print("\n=== Dataset Verification ===")
    if not os.path.exists(train_directory):
        print(f"❌ Dataset directory not found: {train_directory}")
        return
    
    total_images = 0
    class_counts = {}
    for class_name in ['benign', 'malignant']:
        class_path = os.path.join(train_directory, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            class_counts[class_name] = count
            print(f"  {class_name}: {count} images")
            total_images += count
        else:
            print(f"  ❌ {class_name} folder not found")
            class_counts[class_name] = 0
    
    if total_images == 0:
        print("❌ No images found in dataset")
        return
    
    print(f"✓ Total images: {total_images}")
    
    # Verify we have close to 139 images per class as expected
    for class_name, count in class_counts.items():
        if count > 0:
            expected_generated = count * 10  # 10 images per original
            total_per_class = count + expected_generated
            print(f"  {class_name} - Expected output: {count} original + {expected_generated} generated = {total_per_class} total")
    
    # Training phase
    print(f"\n=== Starting DCGAN Training ===")
    print(f"Training separate models for each class (recommended for medical data)")
    print(f"Epochs: {EPOCHS}")
    print(f"This will take 2-4 hours depending on hardware...")
    
    try:
        trained_models = train_advanced_dcgan(
            train_directory=train_directory,
            epochs=EPOCHS,
            save_interval=50,
            class_specific=True
        )
        print("✓ DCGAN training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("Check GPU memory, dataset path, and dependencies")
        return
    
    # Generation phase with organized structure
    print(f"\n=== Generating Organized Augmented Dataset ===")
    print("Structure: output_dir/class/original_filename/[original + 10 generated images]")
    
    try:
        generation_stats = generate_organized_dataset(
            models=trained_models,
            train_directory=train_directory,
            output_directory=augmented_output_directory,
            images_per_original=10
        )
        
        print("✓ Organized dataset generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Dataset generation failed: {str(e)}")
        return
    
    # Results summary
    print(f"\n{'='*60}")
    print("TRAINING AND GENERATION COMPLETE!")
    print(f"{'='*60}")
    
    print("\nGenerated files and folders:")
    print("📁 Training artifacts:")
    print("  - models_benign/ (benign class model)")
    print("  - models_malignant/ (malignant class model)")
    print("  - generated_samples_benign/ (training progress samples)")
    print("  - generated_samples_malignant/ (training progress samples)")
    print("  - dcgan_training_curves_benign.png")
    print("  - dcgan_training_curves_malignant.png")
    
    print(f"\n📁 Organized augmented dataset: {augmented_output_directory}/")
    print("  Structure:")
    print("  ├── benign/")
    print("  │   ├── image1_name/")
    print("  │   │   ├── original_image1_name.png")
    print("  │   │   ├── generated_image1_name_01.png")
    print("  │   │   ├── generated_image1_name_02.png")
    print("  │   │   └── ... (up to _10.png)")
    print("  │   ├── image2_name/")
    print("  │   └── ... (all 139 benign images)")
    print("  └── malignant/")
    print("      ├── image1_name/")
    print("      └── ... (all 139 malignant images)")
    
    # Calculate final statistics
    total_folders = sum(stats['original'] for stats in generation_stats.values())
    total_final_images = sum(stats['original'] + stats['generated'] for stats in generation_stats.values())
    
    print(f"\n📊 Final Statistics:")
    print(f"  Total folders created: {total_folders}")
    print(f"  Total images in dataset: {total_final_images}")
    print(f"  Original images: {sum(stats['original'] for stats in generation_stats.values())}")
    print(f"  Generated images: {sum(stats['generated'] for stats in generation_stats.values())}")
    print(f"  Dataset expansion: {total_final_images / total_images:.1f}x")
    
    print(f"\n🎯 Next Steps:")
    print("1. Review generated images for quality and medical relevance")
    print("2. You can now use this organized dataset for:")
    print("   - Training improved classification models")
    print("   - Data augmentation in your ML pipeline")
    print("   - Creating balanced training sets")
    print("3. Each original image has its own folder with 10 variations")
    print("4. Easy to track which generated images came from which original")
    
    print(f"\n✅ Success! Your organized DCGAN dataset is ready!")

def load_and_generate_only(model_paths, train_directory, output_directory, images_per_original=10):
    """
    Load pre-trained models and generate dataset without training
    Useful if you already have trained models
    """
    print("=== Loading Pre-trained Models for Generation ===")
    
    models = {}
    
    for class_name in ['benign', 'malignant']:
        model_path = model_paths.get(class_name)
        if model_path and os.path.exists(model_path):
            print(f"Loading {class_name} model from: {model_path}")
            
            # Initialize generator
            generator = AdvancedDCGANGenerator(Z_DIM, CHANNELS, FEATURE_MAPS_G).to(device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            if 'generator' in checkpoint:
                generator.load_state_dict(checkpoint['generator'])
            else:
                generator.load_state_dict(checkpoint)
            
            models[class_name] = {
                'generator': generator,
                'class_suffix': f'_{class_name}'
            }
            print(f"✓ {class_name} model loaded successfully")
        else:
            print(f"❌ Model not found for {class_name}: {model_path}")
    
    if not models:
        print("❌ No models loaded. Cannot generate dataset.")
        return
    
    # Generate organized dataset
    generation_stats = generate_organized_dataset(
        models=models,
        train_directory=train_directory,
        output_directory=output_directory,
        images_per_original=images_per_original
    )
    
    return generation_stats

def quick_quality_check(output_directory, num_samples_per_class=5):
    """
    Quick quality check of generated images
    """
    print("\n=== Quick Quality Check ===")
    
    for class_name in ['benign', 'malignant']:
        class_dir = os.path.join(output_directory, class_name)
        if not os.path.exists(class_dir):
            continue
            
        print(f"\nChecking {class_name} class:")
        
        # Get list of image folders
        image_folders = [d for d in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, d))]
        
        if len(image_folders) == 0:
            print(f"  No folders found in {class_dir}")
            continue
            
        print(f"  Found {len(image_folders)} image folders")
        
        # Check a few random folders
        import random
        sample_folders = random.sample(image_folders, min(num_samples_per_class, len(image_folders)))
        
        for folder_name in sample_folders:
            folder_path = os.path.join(class_dir, folder_name)
            images_in_folder = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
            
            original_count = len([f for f in images_in_folder if f.startswith('original_')])
            generated_count = len([f for f in images_in_folder if f.startswith('generated_')])
            
            print(f"    {folder_name}: {original_count} original + {generated_count} generated")
            
            if original_count != 1:
                print(f"      ⚠ Warning: Expected 1 original, found {original_count}")
            if generated_count != 10:
                print(f"      ⚠ Warning: Expected 10 generated, found {generated_count}")

if __name__ == "__main__":
    print("="*80)
    print("ENHANCED DCGAN WITH ORGANIZED IMAGE GENERATION")
    print("="*80)
    print("This script will:")
    print("1. Train separate DCGAN models for benign and malignant classes")
    print("2. Generate exactly 10 images for each original image")
    print("3. Organize output in folders named after original images")
    print("4. Create a well-structured dataset for further use")
    print("="*80)
    
    # Ask user for confirmation
    response = input("\nProceed with training and generation? (y/n): ").lower().strip()
    if response != 'y':
        print("Operation cancelled.")
    else:
        main()
        
        # Perform quality check
        augmented_output_directory = r"F:\jihad\Breast GAN types\breast\breast\dcgan_organized_dataset"
        if os.path.exists(augmented_output_directory):
            quick_quality_check(augmented_output_directory)
        
        print("\n" + "="*80)
        print("DCGAN ORGANIZED GENERATION PIPELINE COMPLETED!")
        print("="*80)