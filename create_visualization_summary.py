"""
Script to create a compressed summary visualization from the CEBaB model visualizations.
"""

import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from io import BytesIO

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create a compressed summary visualization")
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing the visualizations")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file path (default: input_dir/summary_compressed.png)")
    parser.add_argument("--num_examples", type=int, default=3,
                       help="Number of examples to include in the summary")
    parser.add_argument("--quality", type=int, default=85,
                       help="JPEG quality for compression (0-100)")
    parser.add_argument("--max_width", type=int, default=1200,
                       help="Maximum width of the output image")
    
    return parser.parse_args()

def create_summary_image(input_dir, output_file, num_examples=3, quality=85, max_width=1200):
    """Create a summary image from the visualizations"""
    # Find example directories
    example_dirs = [d for d in os.listdir(input_dir) if d.startswith("example_")]
    example_dirs.sort(key=lambda x: int(x.split("_")[1]))
    
    # Limit to the specified number of examples
    example_dirs = example_dirs[:num_examples]
    
    # Load the summary images
    summary_images = []
    for example_dir in example_dirs:
        summary_path = os.path.join(input_dir, example_dir, "summary.png")
        if os.path.exists(summary_path):
            summary_images.append(Image.open(summary_path))
    
    # Calculate the total height
    total_height = sum(img.height for img in summary_images)
    max_width = min(max(img.width for img in summary_images), max_width)
    
    # Create a new image
    combined_image = Image.new("RGB", (max_width, total_height), "white")
    
    # Paste the images
    y_offset = 0
    for img in summary_images:
        # Resize if needed
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)
        
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    # Save the combined image
    if output_file is None:
        output_file = os.path.join(input_dir, "summary_compressed.png")
    
    # Compress the image
    buffer = BytesIO()
    combined_image.save(buffer, format="JPEG", quality=quality)
    compressed_image = Image.open(buffer)
    compressed_image.save(output_file, format="PNG", optimize=True)
    
    print(f"Summary image created: {output_file}")
    print(f"Original size: {combined_image.width}x{combined_image.height}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    
    return output_file

def main():
    args = parse_arguments()
    
    # Create the summary image
    output_file = create_summary_image(
        args.input_dir,
        args.output_file,
        args.num_examples,
        args.quality,
        args.max_width
    )
    
    # Try to open the image
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output_file)}")
    except:
        pass

if __name__ == "__main__":
    main()
