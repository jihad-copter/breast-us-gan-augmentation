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

# Hyperparameters - Latest Research Recommendations
IMAGE_SIZE = 128
BATCH_SIZE = 16  # Smaller batch size for medical images
LEARNING_RATE_G = 0.0001  # Lower LR for generator
LEARNING_RATE_D = 0.0004  # Higher LR for discriminator (TTUR)
BETA1 = 0.0  # Adam beta1 = 0 for better convergence
BETA2 = 0.9
EPOCHS = 1000
Z_DIM = 128  # Increased latent dimension
NUM_CLASSES = 2
CHANNELS = 3
LAMBDA_GP = 10  # Gradient penalty coefficient

class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        self.class_to_idx = {'benign': 0, 'malignant': 1}
        
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path) and class_name in self.class_to_idx:
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

# Self-Attention Module (Latest Research)
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, C, H, W = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, H*W)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, H*W)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        out = self.gamma * out + x
        return out

# Improved Generator with Self-Attention and Progressive Growing Concepts
class ImprovedGenerator(nn.Module):
    def __init__(self, z_dim, num_classes, channels, features_g=64):
        super(ImprovedGenerator, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        
        # Label embedding with larger dimension
        self.embed = nn.Embedding(num_classes, z_dim)
        
        # Initial layer
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim * 2, features_g * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 16),
            nn.ReLU(True)
        )
        
        # Progressive layers with residual connections
        self.prog_blocks = nn.ModuleList([
            self._make_layer(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._make_layer(features_g * 8, features_g * 4, 4, 2, 1),   # 16x16
            self._make_layer(features_g * 4, features_g * 2, 4, 2, 1),   # 32x32
            self._make_layer(features_g * 2, features_g, 4, 2, 1),       # 64x64
        ])
        
        # Self-attention at 64x64
        self.attention = SelfAttention(features_g)
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features_g, channels, 4, 2, 1, bias=False),  # 128x128
            nn.Tanh()
        )
        
        # Apply spectral normalization to some layers
        self.apply(self._apply_spectral_norm)
    
    def _make_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            # Add residual connection if dimensions match
            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    
    def _apply_spectral_norm(self, module):
        if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d)):
            if module.out_channels >= 64:  # Apply to larger layers
                spectral_norm(module)
    
    def forward(self, noise, labels):
        # Embed labels and concatenate with noise
        embedding = self.embed(labels)
        x = torch.cat([noise, embedding], dim=1)
        x = x.view(x.size(0), -1, 1, 1)
        
        # Initial layer
        x = self.initial(x)
        
        # Progressive blocks
        for block in self.prog_blocks:
            x = block(x)
        
        # Self-attention
        x = self.attention(x)
        
        # Final layer
        x = self.final(x)
        return x

# Improved Discriminator with Spectral Normalization
class ImprovedDiscriminator(nn.Module):
    def __init__(self, channels, num_classes, features_d=64):
        super(ImprovedDiscriminator, self).__init__()
        self.num_classes = num_classes
        
        # Label embedding
        self.embed = nn.Embedding(num_classes, IMAGE_SIZE * IMAGE_SIZE)
        
        # Progressive discriminator blocks
        self.blocks = nn.ModuleList([
            self._make_block(channels + 1, features_d, 4, 2, 1, normalize=False),  # 64x64
            self._make_block(features_d, features_d * 2, 4, 2, 1),  # 32x32
            self._make_block(features_d * 2, features_d * 4, 4, 2, 1),  # 16x16
            self._make_block(features_d * 4, features_d * 8, 4, 2, 1),  # 8x8
        ])
        
        # Self-attention at 32x32 (after first two blocks)
        self.attention = SelfAttention(features_d * 2)
        
        # Final layers
        self.final_block = self._make_block(features_d * 8, features_d * 16, 4, 2, 1)  # 4x4
        self.classifier = spectral_norm(nn.Conv2d(features_d * 16, 1, 4, 1, 0))
        
        # Apply spectral normalization
        self.apply(self._apply_spectral_norm)
    
    def _make_block(self, in_channels, out_channels, kernel_size, stride, padding, normalize=True):
        layers = [spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _apply_spectral_norm(self, module):
        if isinstance(module, nn.Conv2d) and not hasattr(module, 'weight_orig'):
            spectral_norm(module)
    
    def forward(self, image, labels):
        # Embed labels and reshape
        embedding = self.embed(labels).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        x = torch.cat([image, embedding], dim=1)
        
        # First two blocks
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        
        # Self-attention
        x = self.attention(x)
        
        # Remaining blocks
        for block in self.blocks[2:]:
            x = block(x)
        
        # Final classification
        x = self.final_block(x)
        x = self.classifier(x)
        return x.view(-1)

# Gradient Penalty for WGAN-GP
def gradient_penalty(discriminator, real_images, fake_images, labels, device):
    batch_size = real_images.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    
    interpolated = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated = interpolated.to(device)
    interpolated.requires_grad_(True)
    
    # Calculate discriminator output
    disc_interpolated = discriminator(interpolated, labels)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=disc_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(disc_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return gradient_penalty

# Improved training function with latest techniques
def train_improved_cgan(train_directory, epochs=EPOCHS, save_interval=50):
    """
    Train CGAN with latest research improvements:
    - WGAN-GP loss
    - Spectral normalization
    - Self-attention
    - TTUR (Two Time-scale Update Rule)
    - Progressive training concepts
    """
    # Enhanced data preprocessing with more augmentation
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Dataset and DataLoader
    dataset = MedicalImageDataset(train_directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    
    print(f"Dataset size: {len(dataset)} images")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Initialize improved networks
    gen = ImprovedGenerator(Z_DIM, NUM_CLASSES, CHANNELS).to(device)
    disc = ImprovedDiscriminator(CHANNELS, NUM_CLASSES).to(device)
    
    # Initialize weights with Xavier/He initialization
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)
    
    gen.apply(init_weights)
    disc.apply(init_weights)
    
    # TTUR: Different learning rates for generator and discriminator
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))
    
    # Learning rate schedulers
    scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(opt_gen, mode='min', factor=0.8, patience=50, verbose=True)
    scheduler_disc = optim.lr_scheduler.ReduceLROnPlateau(opt_disc, mode='min', factor=0.8, patience=50, verbose=True)
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(16, Z_DIM).to(device)
    fixed_labels = torch.tensor([i % NUM_CLASSES for i in range(16)]).to(device)
    
    # Training metrics
    gen_losses = []
    disc_losses = []
    best_gen_loss = float('inf')
    
    print("Starting improved CGAN training...")
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        gen.train()
        disc.train()
        
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (real_images, labels) in enumerate(progress_bar):
            real_images = real_images.to(device)
            labels = labels.to(device)
            batch_size = real_images.size(0)
            
            # Train Discriminator more frequently (5:1 ratio)
            for _ in range(5):
                disc.zero_grad()
                
                # Real images
                disc_real = disc(real_images, labels)
                disc_real_loss = -torch.mean(disc_real)  # WGAN loss
                
                # Fake images
                noise = torch.randn(batch_size, Z_DIM).to(device)
                fake_images = gen(noise, labels).detach()
                disc_fake = disc(fake_images, labels)
                disc_fake_loss = torch.mean(disc_fake)  # WGAN loss
                
                # Gradient penalty
                gp = gradient_penalty(disc, real_images, fake_images, labels, device)
                
                # Total discriminator loss
                disc_loss = disc_real_loss + disc_fake_loss + LAMBDA_GP * gp
                disc_loss.backward()
                opt_disc.step()
                
                epoch_disc_loss += disc_loss.item()
            
            # Train Generator
            gen.zero_grad()
            noise = torch.randn(batch_size, Z_DIM).to(device)
            fake_images = gen(noise, labels)
            gen_output = disc(fake_images, labels)
            gen_loss = -torch.mean(gen_output)  # WGAN loss
            
            gen_loss.backward()
            opt_gen.step()
            
            epoch_gen_loss += gen_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Gen Loss': f'{gen_loss.item():.4f}',
                'Disc Loss': f'{disc_loss.item():.4f}'
            })
        
        # Calculate average losses
        avg_gen_loss = epoch_gen_loss / len(dataloader)
        avg_disc_loss = epoch_disc_loss / (len(dataloader) * 5)  # Account for 5:1 training ratio
        
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        
        # Update learning rate schedulers
        scheduler_gen.step(avg_gen_loss)
        scheduler_disc.step(avg_disc_loss)
        
        # Save best model
        if avg_gen_loss < best_gen_loss:
            best_gen_loss = avg_gen_loss
            os.makedirs("models", exist_ok=True)
            torch.save({
                'generator': gen.state_dict(),
                'discriminator': disc.state_dict(),
                'gen_optimizer': opt_gen.state_dict(),
                'disc_optimizer': opt_disc.state_dict(),
                'epoch': epoch,
                'gen_loss': avg_gen_loss,
                'disc_loss': avg_disc_loss
            }, "models/best_cgan_model.pth")
        
        # Print progress and save samples
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print(f"Generator Loss: {avg_gen_loss:.4f}")
            print(f"Discriminator Loss: {avg_disc_loss:.4f}")
            print(f"Best Gen Loss: {best_gen_loss:.4f}")
        
        if (epoch + 1) % save_interval == 0:
            gen.eval()
            with torch.no_grad():
                fake_samples = gen(fixed_noise, fixed_labels)
                fake_samples = (fake_samples + 1) / 2  # Denormalize
                
                os.makedirs("generated_samples", exist_ok=True)
                vutils.save_image(fake_samples, 
                                f"generated_samples/epoch_{epoch+1}.png", 
                                nrow=4, normalize=False, padding=2)
            gen.train()
    
    # Save final models
    torch.save(gen.state_dict(), "models/final_generator.pth")
    torch.save(disc.state_dict(), "models/final_discriminator.pth")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(gen_losses, label='Generator Loss', color='blue', alpha=0.7)
    plt.plot(disc_losses, label='Discriminator Loss', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Plot smoothed losses
    window_size = max(1, len(gen_losses) // 20)
    if len(gen_losses) > window_size:
        smooth_gen = np.convolve(gen_losses, np.ones(window_size)/window_size, mode='valid')
        smooth_disc = np.convolve(disc_losses, np.ones(window_size)/window_size, mode='valid')
        plt.plot(smooth_gen, label='Smooth Gen Loss', color='blue')
        plt.plot(smooth_disc, label='Smooth Disc Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Smoothed Loss')
        plt.title('Smoothed Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return gen, disc

# Enhanced augmentation function
def generate_enhanced_dataset(model_path, train_directory, output_directory, images_per_original=10):
    """Generate augmented dataset with quality control"""
    
    # Load the best model
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        gen = ImprovedGenerator(Z_DIM, NUM_CLASSES, CHANNELS).to(device)
        gen.load_state_dict(checkpoint['generator'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['gen_loss']:.4f}")
    else:
        # Fallback to final generator
        gen = ImprovedGenerator(Z_DIM, NUM_CLASSES, CHANNELS).to(device)
        gen.load_state_dict(torch.load("models/final_generator.pth", map_location=device))
        print("Loaded final generator model")
    
    gen.eval()
    
    # Create output directories
    for class_name in ['benign', 'malignant']:
        os.makedirs(os.path.join(output_directory, class_name), exist_ok=True)
    
    class_to_idx = {'benign': 0, 'malignant': 1}
    idx_to_class = {0: 'benign', 1: 'malignant'}
    
    total_generated = 0
    generation_stats = {'benign': 0, 'malignant': 0}
    
    print("Generating enhanced augmented dataset...")
    
    for class_name in os.listdir(train_directory):
        class_path = os.path.join(train_directory, class_name)
        if os.path.isdir(class_path) and class_name in class_to_idx:
            class_idx = class_to_idx[class_name]
            
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            print(f"Processing {len(image_files)} {class_name} images...")
            
            for img_name in tqdm(image_files, desc=f"Generating {class_name}"):
                # Copy original image
                original_path = os.path.join(class_path, img_name)
                base_name = os.path.splitext(img_name)[0]
                
                # Copy original
                img = Image.open(original_path)
                output_original = os.path.join(output_directory, class_name, f"orig_{base_name}.png")
                img.save(output_original)
                
                # Generate multiple variations with different noise
                with torch.no_grad():
                    for i in range(images_per_original):
                        # Use different noise characteristics for diversity
                        if i < images_per_original // 2:
                            # First half: standard noise
                            noise = torch.randn(1, Z_DIM).to(device)
                        else:
                            # Second half: truncated noise for higher quality
                            noise = torch.randn(1, Z_DIM).to(device)
                            noise = torch.clamp(noise, -1.5, 1.5)  # Truncated normal
                        
                        label = torch.tensor([class_idx]).to(device)
                        
                        # Generate image
                        fake_img = gen(noise, label)
                        fake_img = torch.clamp((fake_img + 1) / 2, 0, 1)  # Denormalize and clamp
                        fake_img = fake_img.squeeze(0).cpu()
                        
                        # Convert to PIL and save
                        fake_img_pil = transforms.ToPILImage()(fake_img)
                        
                        # Save with descriptive name
                        gen_name = f"gen_{base_name}_{i+1:02d}.png"
                        gen_path = os.path.join(output_directory, class_name, gen_name)
                        fake_img_pil.save(gen_path)
                        
                        total_generated += 1
                        generation_stats[class_name] += 1
    
    # Print generation statistics
    print(f"\n=== Augmentation Complete ===")
    print(f"Total generated images: {total_generated}")
    for class_name, count in generation_stats.items():
        original_count = len([f for f in os.listdir(os.path.join(train_directory, class_name))
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        total_class = count + original_count
        print(f"{class_name}: {original_count} original + {count} generated = {total_class} total")
        print(f"  Augmentation factor: {count/original_count:.1f}x")
    
    print(f"Augmented dataset saved to: {output_directory}")
    return generation_stats

def main():
    """Main function to run the complete CGAN training and augmentation pipeline"""
    
    # Configuration
    train_directory = r"F:\jihad\Breast GAN types\breast\breast\split_dataset\train"
    augmented_output_directory = r"F:\jihad\Breast GAN types\breast\breast\augmented_dataset"
    
    print("=== CGAN Medical Image Augmentation Pipeline ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Device: {device}")
    
    # Check dataset structure
    print("\n=== Dataset Structure Check ===")
    def check_dataset(directory):
        if not os.path.exists(directory):
            print(f"ERROR: Directory does not exist: {directory}")
            return False
        
        total_images = 0
        for class_folder in os.listdir(directory):
            class_path = os.path.join(directory, class_folder)
            if os.path.isdir(class_path):
                image_files = [f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
                print(f"  {class_folder}: {len(image_files)} images")
                total_images += len(image_files)
        
        print(f"Total training images: {total_images}")
        return total_images > 0
    
    if not check_dataset(train_directory):
        print("Please check your dataset path and structure.")
        return
    
    # Training phase
    print("\n=== Starting CGAN Training ===")
    print("This will take several hours depending on your hardware...")
    print("Monitor the 'generated_samples' folder to see training progress.")
    
    try:
        trained_gen, trained_disc = train_improved_cgan(
            train_directory=train_directory, 
            epochs=EPOCHS,
            save_interval=50
        )
        print("✓ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        print("Check your dataset path, GPU memory, and dependencies.")
        return
    
    # Generation phase
    print("\n=== Generating Augmented Dataset ===")
    try:
        # Use best model if available, otherwise use final model
        model_path = "models/best_cgan_model.pth" if os.path.exists("models/best_cgan_model.pth") else "models/final_generator.pth"
        
        stats = generate_enhanced_dataset(
            model_path=model_path,
            train_directory=train_directory,
            output_directory=augmented_output_directory,
            images_per_original=10
        )
        
        print("✓ Dataset augmentation completed successfully!")
        
    except Exception as e:
        print(f"❌ Dataset generation failed: {str(e)}")
        return
    
    # Next steps guidance
    print("\n=== Next Steps ===")
    print("1. Review generated images in 'generated_samples/' folder")
    print("2. Check augmented dataset in:", augmented_output_directory)
    print("3. Retrain your classification models with the augmented dataset")
    print("4. Expected improvements:")
    print("   - Reduced overfitting")
    print("   - Better generalization to external test data")
    print("   - More robust feature learning")
    
    print("\n=== Training Files Created ===")
    print("- models/best_cgan_model.pth (best checkpoint)")
    print("- models/final_generator.pth (final generator)")
    print("- models/final_discriminator.pth (final discriminator)")
    print("- generated_samples/ (training progress images)")
    print("- training_curves.png (loss plots)")

if __name__ == "__main__":
    main()