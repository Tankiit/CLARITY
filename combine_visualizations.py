"""
Script to combine multiple visualization PNG files into a single PNG file.
"""

import os
import argparse
from PIL import Image
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description="Combine multiple PNG files into a single PNG file")
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing the PNG files to combine")
    parser.add_argument("--output_file", type=str, default="combined_visualization.png",
                       help="Output PNG file path")
    parser.add_argument("--max_width", type=int, default=1200,
                       help="Maximum width of the output image")
    parser.add_argument("--quality", type=int, default=85,
                       help="JPEG quality for compression (0-100)")
    parser.add_argument("--files", type=str, nargs="+",
                       default=["concept_activation_counts.png", "concept_cooccurrence.png", 
                                "concept_intervention.png", "concept_tsne_prediction.png"],
                       help="List of PNG files to combine")
    
    return parser.parse_args()

def combine_images(input_dir, output_file, files, max_width=1200, quality=85):
    """Combine multiple PNG files into a single PNG file"""
    # Load images
    images = []
    for file in files:
        file_path = os.path.join(input_dir, file)
        if os.path.exists(file_path):
            images.append(Image.open(file_path))
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not images:
        print("Error: No images found")
        return
    
    # Calculate the total height and maximum width
    total_height = sum(img.height for img in images)
    max_img_width = max(img.width for img in images)
    
    # Resize images if needed
    resized_images = []
    for img in images:
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            resized_images.append(img.resize((max_width, new_height), Image.LANCZOS))
        else:
            # Center the image
            new_img = Image.new("RGB", (max_width, img.height), "white")
            new_img.paste(img, ((max_width - img.width) // 2, 0))
            resized_images.append(new_img)
    
    # Calculate the new total height
    total_height = sum(img.height for img in resized_images)
    
    # Create a new image
    combined_image = Image.new("RGB", (max_width, total_height), "white")
    
    # Paste the images
    y_offset = 0
    for img in resized_images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    # Save the combined image
    combined_image.save(output_file, quality=quality, optimize=True)
    
    print(f"Combined {len(images)} images into {output_file}")
    print(f"Image size: {combined_image.width}x{combined_image.height}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")

def main():
    args = parse_arguments()
    
    # Combine images
    combine_images(args.input_dir, args.output_file, args.files, args.max_width, args.quality)

if __name__ == "__main__":
    main()
