import os
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity
import re
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CycleGANTop5PerOriginal:
    def __init__(self):
        """
        Select top 5 generated images (out of 10) for each original image based on SSIM similarity
        Handles all CycleGAN patterns: A2B, B2A, B2A2B
        """
        pass
        
    def load_and_preprocess_image(self, image_path, target_size=(256, 256)):
        """Load and preprocess image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.array(img)
            return img_array
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def calculate_ssim(self, img1, img2):
        """Calculate SSIM similarity between two images"""
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        return ssim(img1_gray, img2_gray, data_range=255)
    
    def extract_number_from_original(self, filename):
        """Extract number from original filename"""
        match = re.search(r'\((\d+)\)', filename)
        if match:
            return int(match.group(1))
        return None
    
    def find_generated_images_for_original(self, original_number, class_name, generated_dir):
        """
        Find all generated images for a specific original image
        CycleGAN has multiple patterns:
        - cyclegan_enhancement_A2B_malignant (165)_01.png
        - cyclegan_enhancement_B2A_malignant (151)_06.png
        - cyclegan_enhancement_B2A2B_malignant (210)_08.png
        """
        generated_files = []
        
        if not os.path.exists(generated_dir):
            return generated_files
        
        # Multiple patterns for CycleGAN
        patterns = [
            f"cyclegan_enhancement_A2B_{class_name} \({original_number}\)_\d{{2}}\.png",
            f"cyclegan_enhancement_B2A_{class_name} \({original_number}\)_\d{{2}}\.png",
            f"cyclegan_enhancement_B2A2B_{class_name} \({original_number}\)_\d{{2}}\.png"
        ]
        
        for filename in os.listdir(generated_dir):
            for pattern in patterns:
                if re.match(pattern, filename):
                    generated_files.append(filename)
                    break  # Avoid duplicates if somehow multiple patterns match
        
        return sorted(generated_files)
    
    def select_top5_for_each_original(self, original_dirs, cyclegan_dirs, output_dirs):
        """Select top 5 generated images for each original based on SSIM similarity"""
        print(f"\n{'='*80}")
        print("SELECTING TOP 5 GENERATED IMAGES FOR EACH ORIGINAL (CYCLEGAN)")
        print("Handling patterns: A2B, B2A, B2A2B")
        print(f"{'='*80}")
        
        total_originals = 0
        total_processed = 0
        total_generated_moved = 0
        summary_results = []
        pattern_stats = {'A2B': 0, 'B2A': 0, 'B2A2B': 0}
        
        for class_name in ['malignant', 'benign']:
            original_dir = original_dirs[class_name]
            generated_dir = cyclegan_dirs[class_name]
            output_dir = output_dirs[class_name]
            
            if not os.path.exists(original_dir) or not os.path.exists(generated_dir):
                print(f"Warning: Directory not found for {class_name}")
                continue
            
            print(f"\nProcessing {class_name} images...")
            
            # Get original files
            original_files = [f for f in os.listdir(original_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Found {len(original_files)} original {class_name} images")
            total_originals += len(original_files)
            
            class_moved_count = 0
            class_pattern_stats = {'A2B': 0, 'B2A': 0, 'B2A2B': 0}
            
            for i, original_filename in enumerate(original_files):
                # Extract number from original filename
                original_number = self.extract_number_from_original(original_filename)
                
                if original_number is None:
                    print(f"  Could not extract number from {original_filename}")
                    continue
                
                # Find generated images for this original
                generated_files = self.find_generated_images_for_original(
                    original_number, class_name, generated_dir
                )
                
                if len(generated_files) == 0:
                    print(f"  No generated images found for {original_filename}")
                    continue
                
                # Load original image
                original_path = os.path.join(original_dir, original_filename)
                original_img = self.load_and_preprocess_image(original_path)
                
                if original_img is None:
                    continue
                
                # Calculate similarity for each generated image
                similarities = []
                
                for gen_filename in generated_files:
                    gen_path = os.path.join(generated_dir, gen_filename)
                    gen_img = self.load_and_preprocess_image(gen_path)
                    
                    if gen_img is not None:
                        similarity_score = self.calculate_ssim(original_img, gen_img)
                        
                        # Determine pattern type
                        pattern_type = 'A2B'
                        if 'B2A2B' in gen_filename:
                            pattern_type = 'B2A2B'
                        elif 'B2A' in gen_filename:
                            pattern_type = 'B2A'
                        
                        similarities.append({
                            'filename': gen_filename,
                            'path': gen_path,
                            'similarity': similarity_score,
                            'pattern_type': pattern_type
                        })
                
                if len(similarities) == 0:
                    print(f"  No valid generated images for {original_filename}")
                    continue
                
                # Sort by similarity (descending) and select top 5
                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                top5_similarities = similarities[:5]
                
                # Copy top 5 generated images
                copied_count = 0
                for rank, sim_data in enumerate(top5_similarities, 1):
                    src_path = sim_data['path']
                    similarity_score = sim_data['similarity']
                    pattern_type = sim_data['pattern_type']
                    
                    # Create new filename with similarity score, rank, and pattern type
                    base_name = os.path.splitext(sim_data['filename'])[0]
                    new_filename = f"{base_name}_rank{rank}_sim{similarity_score:.4f}_{pattern_type}.png"
                    dst_path = os.path.join(output_dir, new_filename)
                    
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                        class_pattern_stats[pattern_type] += 1
                        pattern_stats[pattern_type] += 1
                    except Exception as e:
                        print(f"    Error copying {sim_data['filename']}: {e}")
                
                if copied_count > 0:
                    total_processed += 1
                    class_moved_count += copied_count
                    
                    # Store summary data
                    pattern_distribution = {
                        'A2B_count': sum(1 for s in top5_similarities if s['pattern_type'] == 'A2B'),
                        'B2A_count': sum(1 for s in top5_similarities if s['pattern_type'] == 'B2A'),
                        'B2A2B_count': sum(1 for s in top5_similarities if s['pattern_type'] == 'B2A2B')
                    }
                    
                    summary_results.append({
                        'Original_File': original_filename,
                        'Original_Number': original_number,
                        'Class': class_name,
                        'Total_Generated_Found': len(similarities),
                        'Top5_Moved': copied_count,
                        'Avg_Top5_Similarity_%': round(np.mean([s['similarity'] for s in top5_similarities]) * 100, 2),
                        'Best_Similarity_%': round(top5_similarities[0]['similarity'] * 100, 2),
                        'Worst_Top5_Similarity_%': round(top5_similarities[-1]['similarity'] * 100, 2),
                        'A2B_in_Top5': pattern_distribution['A2B_count'],
                        'B2A_in_Top5': pattern_distribution['B2A_count'],
                        'B2A2B_in_Top5': pattern_distribution['B2A2B_count']
                    })
                
                # Progress update
                if (i + 1) % 10 == 0 or (i + 1) == len(original_files):
                    print(f"  Processed {i + 1}/{len(original_files)} {class_name} images...")
            
            total_generated_moved += class_moved_count
            print(f"  {class_name.upper()} SUMMARY:")
            print(f"    Images processed: {len([r for r in summary_results if r['Class'] == class_name])}")
            print(f"    Generated images moved: {class_moved_count}")
            print(f"    Pattern distribution: A2B={class_pattern_stats['A2B']}, B2A={class_pattern_stats['B2A']}, B2A2B={class_pattern_stats['B2A2B']}")
        
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"Total original images found: {total_originals}")
        print(f"Total original images processed: {total_processed}")
        print(f"Total generated images moved: {total_generated_moved}")
        print(f"Expected total (if all successful): {total_processed * 5}")
        print(f"\n📊 PATTERN DISTRIBUTION IN TOP 5 SELECTIONS:")
        print(f"  A2B patterns: {pattern_stats['A2B']} ({pattern_stats['A2B']/max(1,total_generated_moved)*100:.1f}%)")
        print(f"  B2A patterns: {pattern_stats['B2A']} ({pattern_stats['B2A']/max(1,total_generated_moved)*100:.1f}%)")
        print(f"  B2A2B patterns: {pattern_stats['B2A2B']} ({pattern_stats['B2A2B']/max(1,total_generated_moved)*100:.1f}%)")
        
        return summary_results, pattern_stats
    
    def create_output_directories(self, base_output_dir):
        """Create output directory structure"""
        output_dirs = {}
        
        for class_name in ['malignant', 'benign']:
            class_dir = Path(base_output_dir) / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            output_dirs[class_name] = str(class_dir)
            print(f"✅ Created output directory: {class_dir}")
        
        return output_dirs
    
    def save_summary_report(self, summary_results, pattern_stats, output_base_dir):
        """Save summary report to CSV"""
        if not summary_results:
            print("No summary data to save.")
            return None
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_results)
        
        # Add overall statistics
        total_moved = summary_df['Top5_Moved'].sum()
        avg_similarity = summary_df['Avg_Top5_Similarity_%'].mean()
        
        # Save detailed CSV
        summary_file = Path(output_base_dir) / "CycleGAN_Top5_Selection_Summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Create and save overall statistics
        stats_data = [{
            'Total_Original_Images': len(summary_df),
            'Total_Generated_Moved': total_moved,
            'Expected_Total': len(summary_df) * 5,
            'Success_Rate_%': round((total_moved / (len(summary_df) * 5)) * 100, 2) if len(summary_df) > 0 else 0,
            'Overall_Avg_Top5_Similarity_%': round(avg_similarity, 2),
            'Malignant_Count': len(summary_df[summary_df['Class'] == 'malignant']),
            'Benign_Count': len(summary_df[summary_df['Class'] == 'benign']),
            'A2B_Selected': pattern_stats['A2B'],
            'B2A_Selected': pattern_stats['B2A'],
            'B2A2B_Selected': pattern_stats['B2A2B'],
            'A2B_Percentage': round(pattern_stats['A2B']/max(1,total_moved)*100, 2),
            'B2A_Percentage': round(pattern_stats['B2A']/max(1,total_moved)*100, 2),
            'B2A2B_Percentage': round(pattern_stats['B2A2B']/max(1,total_moved)*100, 2)
        }]
        
        stats_df = pd.DataFrame(stats_data)
        stats_file = Path(output_base_dir) / "CycleGAN_Top5_Overall_Statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        
        # Create pattern analysis CSV
        pattern_analysis = []
        for index, row in summary_df.iterrows():
            pattern_analysis.append({
                'Original_File': row['Original_File'],
                'Class': row['Class'],
                'Best_Pattern_in_Top5': 'A2B' if row['A2B_in_Top5'] >= max(row['B2A_in_Top5'], row['B2A2B_in_Top5']) 
                                      else ('B2A' if row['B2A_in_Top5'] >= row['B2A2B_in_Top5'] else 'B2A2B'),
                'A2B_in_Top5': row['A2B_in_Top5'],
                'B2A_in_Top5': row['B2A_in_Top5'],
                'B2A2B_in_Top5': row['B2A2B_in_Top5'],
                'Best_Similarity_%': row['Best_Similarity_%']
            })
        
        pattern_df = pd.DataFrame(pattern_analysis)
        pattern_file = Path(output_base_dir) / "CycleGAN_Top5_Pattern_Analysis.csv"
        pattern_df.to_csv(pattern_file, index=False)
        
        print(f"\n✅ Summary report saved: {summary_file}")
        print(f"✅ Statistics saved: {stats_file}")
        print(f"✅ Pattern analysis saved: {pattern_file}")
        
        # Display key statistics
        print(f"\n📊 KEY STATISTICS:")
        print(f"  Total original images processed: {len(summary_df)}")
        print(f"  Total generated images moved: {total_moved}")
        print(f"  Expected total (5 per original): {len(summary_df) * 5}")
        print(f"  Success rate: {stats_data[0]['Success_Rate_%']}%")
        print(f"  Average top-5 similarity: {avg_similarity:.2f}%")
        print(f"\n🔄 CYCLEGAN PATTERN PERFORMANCE:")
        print(f"  A2B selected: {pattern_stats['A2B']} ({stats_data[0]['A2B_Percentage']}%)")
        print(f"  B2A selected: {pattern_stats['B2A']} ({stats_data[0]['B2A_Percentage']}%)")
        print(f"  B2A2B selected: {pattern_stats['B2A2B']} ({stats_data[0]['B2A2B_Percentage']}%)")
        
        return summary_file, stats_file, pattern_file

def main():
    """Main function to select top 5 generated images per original for CycleGAN"""
    
    # Define directories
    original_directories = {
        'malignant': r"F:\jihad\Breast GAN types\breast\breast\split_dataset\train\malignant",
        'benign': r"F:\jihad\Breast GAN types\breast\breast\split_dataset\train\benign"
    }
    
    cyclegan_directories = {
        'malignant': r"F:\jihad\Breast GAN types\breast\breast\cyclegan_augmented\generate data\malignant",
        'benign': r"F:\jihad\Breast GAN types\breast\breast\cyclegan_augmented\generate data\benign"
    }
    
    # Output directory - this will contain only the top 5 generated images per original
    output_base_directory = r"F:\jihad\Breast GAN types\breast\breast\cyclegan_top5_generated"
    
    # Initialize selector
    selector = CycleGANTop5PerOriginal()
    
    print("🚀 Starting CycleGAN Top 5 Selection Process...")
    print("📋 Goal: Select top 5 generated images (out of 10) for each original image")
    print("🔄 Handling all CycleGAN patterns: A2B, B2A, B2A2B")
    print(f"📁 Output directory: {output_base_directory}")
    
    # Create output directories
    print("\n📁 Creating output directory structure...")
    output_dirs = selector.create_output_directories(output_base_directory)
    
    # Select and move top 5 images for each original
    print("\n🔍 Processing all original images...")
    summary_results, pattern_stats = selector.select_top5_for_each_original(
        original_directories,
        cyclegan_directories,
        output_dirs
    )
    
    # Save summary report
    print("\n💾 Saving summary reports...")
    summary_file, stats_file, pattern_file = selector.save_summary_report(
        summary_results, pattern_stats, output_base_directory
    )
    
    print(f"\n{'='*80}")
    print("✅ CYCLEGAN TOP 5 SELECTION COMPLETE!")
    print(f"{'='*80}")
    print(f"📊 Check the output directory: {output_base_directory}")
    print(f"📋 Summary report: {summary_file}")
    print(f"📈 Statistics report: {stats_file}")
    print(f"🔄 Pattern analysis: {pattern_file}")
    
    return summary_results, pattern_stats

if __name__ == "__main__":
    results, patterns = main()