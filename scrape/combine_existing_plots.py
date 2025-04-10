#!/usr/bin/env python3
"""
Combine existing scatter plot images into a single figure.
This script takes the three separate model vs source rating plots and
combines them into one figure for easy comparison.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Create output directory if it doesn't exist
OUTPUT_DIR = "figures/combined"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define input image paths
DEEPSEEK_PLOT = "figures/balanced_dataset/deepseek_vs_source_ratings.png"
OPENAI_PLOT = "figures/balanced_dataset/openai_vs_source_ratings.png"
GEMINI_PLOT = "figures/balanced_dataset/gemini_vs_source_ratings.png"

def combine_plots():
    """Combine the three model plots into a single figure"""
    # Verify all input files exist
    for filepath in [DEEPSEEK_PLOT, OPENAI_PLOT, GEMINI_PLOT]:
        if not os.path.exists(filepath):
            print(f"Error: Input file {filepath} not found")
            return False
    
    # Create figure with subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    # Load and display each image in its subplot
    images = [DEEPSEEK_PLOT, OPENAI_PLOT, GEMINI_PLOT]
    titles = ["DeepSeek V3 vs. Source Ratings", 
              "OpenAI GPT-4o vs. Source Ratings", 
              "Google Gemini vs. Source Ratings"]
    
    for i, (img_path, title) in enumerate(zip(images, titles)):
        img = imread(img_path)
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        axes[i].axis('off')  # Hide axes
    
    # No main title (removed as requested)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the combined figure
    output_path = os.path.join(OUTPUT_DIR, "combined_model_comparisons.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined figure to {output_path}")
    
    return True

def main():
    """Main function"""
    combine_plots()

if __name__ == "__main__":
    main() 