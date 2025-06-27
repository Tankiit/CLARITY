"""
Script to combine multiple PDFs into a single PDF file.
"""

import os
import argparse
from PyPDF2 import PdfMerger

def parse_arguments():
    parser = argparse.ArgumentParser(description="Combine multiple PDFs into a single PDF file")
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing the PDFs to combine")
    parser.add_argument("--output_file", type=str, default="combined.pdf",
                       help="Output PDF file path")
    parser.add_argument("--file_pattern", type=str, default="combined.pdf",
                       help="Pattern to match PDF files (e.g., 'combined.pdf' or '*.pdf')")
    
    return parser.parse_args()

def combine_pdfs(input_dir, output_file, file_pattern):
    """Combine multiple PDFs into a single PDF file"""
    # Find all PDF files matching the pattern
    pdf_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == file_pattern or (file_pattern == "*.pdf" and file.endswith(".pdf")):
                pdf_files.append(os.path.join(root, file))
    
    # Sort the PDF files by directory name
    pdf_files.sort()
    
    # Create a PDF merger
    merger = PdfMerger()
    
    # Add each PDF to the merger
    for pdf_file in pdf_files:
        print(f"Adding {pdf_file}")
        merger.append(pdf_file)
    
    # Write the combined PDF to the output file
    print(f"Writing combined PDF to {output_file}")
    merger.write(output_file)
    merger.close()
    
    print(f"Combined {len(pdf_files)} PDFs into {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")

def main():
    args = parse_arguments()
    
    # Combine PDFs
    combine_pdfs(args.input_dir, args.output_file, args.file_pattern)

if __name__ == "__main__":
    main()
