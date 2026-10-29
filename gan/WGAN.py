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
from pathlib import Path
import json

warnings.filterwarnings("ignore")

# Global Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Hyperparameters
CONFIG = {
    'image_size': 128,
    'batch_size': 16,
    'z_dim': 128,
    'channels': 3,
    'epochs': 1000,
    'lr_g': 0.0001,
    'lr_d': 0.0004,
    'beta1': 0.0,
    'beta2': 0.9,
    'lambda_gp': 10,
    'n_critic': 5,
    'save_interval': 10,
    'images_per_original': 10
}

class MedicalDataset(Dataset):
    """Simple, robust dataset class"""
    
    def __init__(self, root_dir, class_filter=None, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load images
        classes = ['benign', 'malignant']
        for i, class_name in enumerate(classes):
            if class_filter and class_name != class_filter:
                continue
                
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
                
            for img_path in class_dir.glob("*.png"):
                self.images.append(str(img_path))
                self.labels.append(i)
            for img_path in class_dir.glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(i)
            for img_path in class_dir.glob("*.jpeg"):
                self.images.append(str(img_path))
                self.labels.append(i)
        
        print(f"Loaded {len(self.images)} images for {class_filter or 'all classes'}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random other image
            new_idx = np.random.randint(0, len(self.images))
            return self.__getitem__(new_idx)

class Generator(nn.Module):
    """Simple but effective generator"""
    
    def __init__(self, z_dim=128, channels=3, features=64):
        super(Generator, self).__init__()
        
        # Initial layer: z_dim -> 4x4x(features*16)
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim, features * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(True)
        )
        
        # Upsampling layers
        self.layer1 = self._make_layer(features * 16, features * 8)  # 4x4 -> 8x8
        self.layer2 = self._make_layer(features * 8, features * 4)   # 8x8 -> 16x16
        self.layer3 = self._make_layer(features * 4, features * 2)   # 16x16 -> 32x32
        self.layer4 = self._make_layer(features * 2, features)       # 32x32 -> 64x64
        
        # Final layer: 64x64 -> 128x128
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self._initialize_weights()
    
    def _make_layer(self, in_features, out_features):
        return nn.Sequential(
            nn.ConvTranspose2d(in_features, out_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)
    
    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        x = self.initial(z)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        return x

class Discriminator(nn.Module):
    """WGAN-GP Discriminator with spectral normalization"""
    
    def __init__(self, channels=3, features=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # 128x128 -> 64x64
            spectral_norm(nn.Conv2d(channels, features, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64x64 -> 32x32
            spectral_norm(nn.Conv2d(features, features * 2, 4, 2, 1)),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 -> 16x16
            spectral_norm(nn.Conv2d(features * 2, features * 4, 4, 2, 1)),
            nn.InstanceNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 -> 8x8
            spectral_norm(nn.Conv2d(features * 4, features * 8, 4, 2, 1)),
            nn.InstanceNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 -> 4x4
            spectral_norm(nn.Conv2d(features * 8, features * 16, 4, 2, 1)),
            nn.InstanceNorm2d(features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 -> 1x1
            spectral_norm(nn.Conv2d(features * 16, 1, 4, 1, 0))
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
    
    def forward(self, x):
        return self.main(x).view(-1)

def gradient_penalty(discriminator, real_samples, fake_samples, device):
    """Calculate gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # Interpolate between real and fake samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated = interpolated.to(device)
    interpolated.requires_grad_(True)
    
    # Calculate discriminator output for interpolated samples
    d_interpolated = discriminator(interpolated)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    
    return penalty

def train_wgan_gp(data_dir, output_dir, class_name=None):
    """Train WGAN-GP model"""
    
    print(f"\n=== Training WGAN-GP for {class_name or 'all classes'} ===")
    
    # Create output directories
    output_path = Path(output_dir)
    model_dir = output_path / f"models_{class_name or 'all'}"
    sample_dir = output_path / f"samples_{class_name or 'all'}"
    model_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Dataset and dataloader
    dataset = MedicalDataset(data_dir, class_filter=class_name, transform=transform)
    if len(dataset) < 10:
        raise ValueError(f"Not enough images for training: {len(dataset)}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=2,
        drop_last=True
    )
    
    # Initialize networks
    generator = Generator(CONFIG['z_dim'], CONFIG['channels']).to(DEVICE)
    discriminator = Discriminator(CONFIG['channels']).to(DEVICE)
    
    # Optimizers
    opt_gen = optim.Adam(generator.parameters(), lr=CONFIG['lr_g'], betas=(CONFIG['beta1'], CONFIG['beta2']))
    opt_disc = optim.Adam(discriminator.parameters(), lr=CONFIG['lr_d'], betas=(CONFIG['beta1'], CONFIG['beta2']))
    
    # Training metrics
    g_losses = []
    d_losses = []
    wasserstein_distances = []
    
    # Fixed noise for monitoring
    fixed_noise = torch.randn(16, CONFIG['z_dim']).to(DEVICE)
    
    print(f"Starting training with {len(dataset)} images, {len(dataloader)} batches per epoch")
    
    # Training loop
    for epoch in range(CONFIG['epochs']):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_wd = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for batch_idx, (real_images, _) in enumerate(pbar):
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)
            
            # Train Discriminator
            for _ in range(CONFIG['n_critic']):
                discriminator.zero_grad()
                
                # Real images
                d_real = discriminator(real_images)
                
                # Fake images
                noise = torch.randn(batch_size, CONFIG['z_dim']).to(DEVICE)
                fake_images = generator(noise).detach()
                d_fake = discriminator(fake_images)
                
                # Gradient penalty
                gp = gradient_penalty(discriminator, real_images, fake_images, DEVICE)
                
                # Discriminator loss
                d_loss = torch.mean(d_fake) - torch.mean(d_real) + CONFIG['lambda_gp'] * gp
                d_loss.backward()
                opt_disc.step()
                
                epoch_d_loss += d_loss.item()
                epoch_wd += (torch.mean(d_real) - torch.mean(d_fake)).item()
            
            # Train Generator
            generator.zero_grad()
            
            noise = torch.randn(batch_size, CONFIG['z_dim']).to(DEVICE)
            fake_images = generator(noise)
            d_fake = discriminator(fake_images)
            
            g_loss = -torch.mean(d_fake)
            g_loss.backward()
            opt_gen.step()
            
            epoch_g_loss += g_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G_Loss': f"{g_loss.item():.4f}",
                'D_Loss': f"{d_loss.item():.4f}",
                'WD': f"{(torch.mean(d_real) - torch.mean(d_fake)).item():.4f}"
            })
        
        # Calculate averages
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / (len(dataloader) * CONFIG['n_critic'])
        avg_wd = epoch_wd / (len(dataloader) * CONFIG['n_critic'])
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        wasserstein_distances.append(avg_wd)
        
        # Save samples
        if (epoch + 1) % CONFIG['save_interval'] == 0:
            generator.eval()
            with torch.no_grad():
                fake_samples = generator(fixed_noise)
                fake_samples = (fake_samples + 1) / 2  # Denormalize
                vutils.save_image(
                    fake_samples, 
                    sample_dir / f"epoch_{epoch+1:03d}.png", 
                    nrow=4, normalize=False, padding=2
                )
            generator.train()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: G_Loss={avg_g_loss:.4f}, D_Loss={avg_d_loss:.4f}, WD={avg_wd:.4f}")
    
    # Save final models
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'opt_gen': opt_gen.state_dict(),
        'opt_disc': opt_disc.state_dict(),
        'config': CONFIG,
        'losses': {'g_losses': g_losses, 'd_losses': d_losses, 'wd': wasserstein_distances}
    }, model_dir / 'final_model.pth')
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(g_losses, label='Generator')
    plt.plot(d_losses, label='Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(wasserstein_distances)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Wasserstein Distance')
    plt.title('Wasserstein Distance')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    if len(g_losses) > 20:
        smooth_g = np.convolve(g_losses, np.ones(10)/10, mode='valid')
        smooth_d = np.convolve(d_losses, np.ones(10)/10, mode='valid')
        plt.plot(smooth_g, label='Generator (smooth)')
        plt.plot(smooth_d, label='Discriminator (smooth)')
        plt.xlabel('Epoch')
        plt.ylabel('Smoothed Loss')
        plt.title('Loss Convergence')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / f'training_curves_{class_name or "all"}.png', dpi=300)
    plt.close()
    
    print(f"Training completed! Models saved to {model_dir}")
    
    return {
        'generator': generator,
        'discriminator': discriminator,
        'losses': {'g_losses': g_losses, 'd_losses': d_losses, 'wd': wasserstein_distances}
    }

def load_trained_model(model_path):
    """Load a trained model"""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    generator = Generator(CONFIG['z_dim'], CONFIG['channels']).to(DEVICE)
    discriminator = Discriminator(CONFIG['channels']).to(DEVICE)
    
    generator.load_state_dict(checkpoint['generator'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    
    return {
        'generator': generator,
        'discriminator': discriminator,
        'losses': checkpoint.get('losses', {})
    }

def test_generation(generator, output_dir, num_samples=10):
    """Test image generation"""
    
    os.makedirs(output_dir, exist_ok=True)
    generator.eval()
    
    print(f"Testing generation of {num_samples} images...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Generate noise
            noise = torch.randn(1, CONFIG['z_dim']).to(DEVICE)
            
            # Generate image
            fake_img = generator(noise)
            
            # Process and save
            fake_img = torch.clamp((fake_img + 1) / 2, 0, 1)
            fake_img_pil = transforms.ToPILImage()(fake_img.squeeze(0).cpu())
            
            save_path = os.path.join(output_dir, f"test_gen_{i+1:03d}.png")
            fake_img_pil.save(save_path)
            
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                print(f"Generated {save_path} ({file_size} bytes)")
            else:
                print(f"Failed to save {save_path}")
    
    generator.train()
    print("Test generation completed!")

def generate_augmented_dataset(models, input_dir, output_dir):
    """Generate augmented dataset with guaranteed image creation"""
    
    print("=== Generating Augmented Dataset ===")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output structure
    for class_name in ['benign', 'malignant']:
        class_dir = output_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {class_dir}")
    
    generation_stats = {'benign': 0, 'malignant': 0}
    
    for class_name in ['benign', 'malignant']:
        print(f"\n--- Processing {class_name} class ---")
        
        class_input_dir = input_path / class_name
        class_output_dir = output_path / class_name
        
        if not class_input_dir.exists():
            print(f"Input directory not found: {class_input_dir}")
            continue
        
        # Get generator
        generator = None
        if class_name in models:
            generator = models[class_name]['generator']
            print(f"Using class-specific generator")
        elif 'all' in models:
            generator = models['all']['generator']
            print(f"Using general generator")
        else:
            print(f"No generator available for {class_name}")
            continue
        
        generator.eval()
        
        # Get original images
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(list(class_input_dir.glob(ext)))
        
        print(f"Found {len(image_files)} original images")
        
        if len(image_files) == 0:
            continue
        
        # Process each original image
        for img_idx, img_file in enumerate(tqdm(image_files, desc=f"Generating {class_name}")):
            
            try:
                # Load and save original
                original_img = Image.open(img_file).convert('RGB')
                original_img = original_img.resize((CONFIG['image_size'], CONFIG['image_size']), Image.LANCZOS)
                
                base_name = img_file.stem
                orig_path = class_output_dir / f"orig_{base_name}.png"
                original_img.save(orig_path)
                
                # Generate synthetic images
                generated_count = 0
                max_attempts = CONFIG['images_per_original'] * 2
                
                with torch.no_grad():
                    for attempt in range(max_attempts):
                        if generated_count >= CONFIG['images_per_original']:
                            break
                        
                        try:
                            # Generate image
                            noise = torch.randn(1, CONFIG['z_dim']).to(DEVICE)
                            fake_img = generator(noise)
                            
                            # Process image
                            fake_img = torch.clamp((fake_img + 1) / 2, 0, 1)
                            fake_img_tensor = fake_img.squeeze(0).cpu()
                            
                            # Basic quality check
                            img_var = torch.var(fake_img_tensor).item()
                            img_mean = torch.mean(fake_img_tensor).item()
                            
                            if img_var > 0.005 and 0.05 < img_mean < 0.95:
                                # Save image
                                fake_img_pil = transforms.ToPILImage()(fake_img_tensor)
                                gen_path = class_output_dir / f"gen_{base_name}_{generated_count+1:03d}.png"
                                fake_img_pil.save(gen_path)
                                
                                # Verify save
                                if gen_path.exists() and gen_path.stat().st_size > 1000:
                                    generated_count += 1
                                    generation_stats[class_name] += 1
                                    
                                    if generated_count % 5 == 0:
                                        print(f"  Generated {generated_count}/{CONFIG['images_per_original']} for {img_file.name}")
                        
                        except Exception as e:
                            print(f"  Generation error: {e}")
                            continue
                
                if generated_count < CONFIG['images_per_original']:
                    print(f"  Warning: Only generated {generated_count}/{CONFIG['images_per_original']} for {img_file.name}")
                
            except Exception as e:
                print(f"  Error processing {img_file}: {e}")
                continue
        
        generator.train()
    
    # Print final statistics
    print(f"\n=== Generation Complete ===")
    print(f"Generated images:")
    for class_name, count in generation_stats.items():
        class_dir = output_path / class_name
        actual_files = len(list(class_dir.glob("*.png")))
        print(f"  {class_name}: {count} reported, {actual_files} files on disk")
        
        # Show sample files
        sample_files = list(class_dir.glob("gen_*.png"))[:3]
        if sample_files:
            print(f"    Sample generated files: {[f.name for f in sample_files]}")
    
    return generation_stats

def main():
    """Main function"""
    
    # Configuration
    input_dir = r"F:\jihad\Breast GAN types\breast\breast\split_dataset\train"
    output_dir = r"F:\jihad\Breast GAN types\breast\breast\wgan_output"
    
    print("="*80)
    print("WGAN-GP MEDICAL IMAGE AUGMENTATION")
    print("="*80)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    try:
        # Validate input directory
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Error: Input directory not found: {input_dir}")
            print("Please check your path and ensure the dataset exists with structure:")
            print("  dataset/")
            print("  ├── benign/")
            print("  │   ├── image1.jpg")
            print("  │   └── ...")
            print("  └── malignant/")
            print("      ├── image1.jpg")
            print("      └── ...")
            return
        
        # Check dataset
        total_images = 0
        for class_name in ['benign', 'malignant']:
            class_dir = input_path / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.png"))) + len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.jpeg")))
                print(f"{class_name}: {count} images")
                total_images += count
            else:
                print(f"{class_name}: directory not found")
        
        if total_images == 0:
            print("Error: No images found in dataset")
            return
        
        print(f"Total images: {total_images}")
        
        # Train models
        print("\n" + "="*50)
        print("TRAINING PHASE")
        print("="*50)
        
        trained_models = {}
        
        # Train class-specific models
        for class_name in ['benign', 'malignant']:
            class_dir = input_path / class_name
            if class_dir.exists():
                image_count = len(list(class_dir.glob("*.png"))) + len(list(class_dir.glob("*.jpg")))
                
                if image_count >= 20:  # Minimum for training
                    print(f"\nTraining model for {class_name} ({image_count} images)...")
                    
                    try:
                        model = train_wgan_gp(
                            data_dir=input_dir,
                            output_dir=output_dir,
                            class_name=class_name
                        )
                        trained_models[class_name] = model
                        print(f"Successfully trained {class_name} model")
                        
                        # Test generation immediately after training
                        print(f"Testing {class_name} generator...")
                        test_generation(
                            generator=model['generator'],
                            output_dir=os.path.join(output_dir, f"test_{class_name}"),
                            num_samples=5
                        )
                        
                    except Exception as e:
                        print(f"Failed to train {class_name} model: {e}")
                else:
                    print(f"Skipping {class_name}: only {image_count} images (need at least 20)")
        
        if not trained_models:
            print("Error: No models were successfully trained")
            return
        
        # Generate augmented dataset
        print("\n" + "="*50)
        print("GENERATION PHASE")
        print("="*50)
        
        augmented_dir = os.path.join(output_dir, "augmented_dataset")
        generation_stats = generate_augmented_dataset(
            models=trained_models,
            input_dir=input_dir,
            output_dir=augmented_dir
        )
        
        # Final summary
        print("\n" + "="*80)
        print("COMPLETION SUMMARY")
        print("="*80)
        
        print(f"Trained models: {list(trained_models.keys())}")
        print(f"Output directory: {output_dir}")
        print(f"Augmented dataset: {augmented_dir}")
        
        total_generated = sum(generation_stats.values())
        print(f"\nGeneration results:")
        print(f"  Total original images: {total_images}")
        print(f"  Total generated images: {total_generated}")
        print(f"  Benign generated: {generation_stats.get('benign', 0)}")
        print(f"  Malignant generated: {generation_stats.get('malignant', 0)}")
        
        if total_images > 0:
            augmentation_factor = (total_images + total_generated) / total_images
            print(f"  Augmentation factor: {augmentation_factor:.1f}x")
        
        # Verify final output
        aug_path = Path(augmented_dir)
        if aug_path.exists():
            print(f"\nFinal verification:")
            for class_name in ['benign', 'malignant']:
                class_dir = aug_path / class_name
                if class_dir.exists():
                    files = list(class_dir.glob("*.png"))
                    orig_files = [f for f in files if f.name.startswith('orig_')]
                    gen_files = [f for f in files if f.name.startswith('gen_')]
                    print(f"  {class_name}: {len(orig_files)} originals + {len(gen_files)} generated = {len(files)} total")
        
        print("\nWGAN-GP training and augmentation completed successfully!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()