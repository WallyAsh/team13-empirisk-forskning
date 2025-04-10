#!/usr/bin/env python3
"""
Create an error distribution histogram comparing all three AI models.
This script generates a visualization showing how errors are distributed
across the models, with background shading for error categories.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# Create figures directory if it doesn't exist
OUTPUT_DIR = "figures/publication"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths for the AI model results
DEEPSEEK_FILE = 'balanced_dataset/cleaned_articles_rated_deepseek.json'
OPENAI_FILE = 'balanced_dataset/cleaned_articles_rated_openai.json'
GEMINI_FILE = 'balanced_dataset/cleaned_articles_rated_gemini.json'

# Model configurations with colors that match the example
MODEL_CONFIGS = [
    {'name': 'DeepSeek', 'file': DEEPSEEK_FILE, 'key': 'deepseek', 'color': '#1E88E5', 'hatch': ''},  # Deeper blue
    {'name': 'OpenAI GPT-4o', 'file': OPENAI_FILE, 'key': 'openai', 'color': '#D81B60', 'hatch': ''},  # Magenta/pink
    {'name': 'Google Gemini', 'file': GEMINI_FILE, 'key': 'gemini', 'color': '#FFC107', 'hatch': ''}  # Yellow/gold
]

def get_category_from_rating(rating):
    """Convert numerical rating to category string"""
    if rating is None:
        return None
    if rating <= -3:
        return "Left"
    elif rating < -1:
        return "Lean Left"
    elif rating <= 1:
        return "Center"
    elif rating < 3:
        return "Lean Right"
    else:
        return "Right"

def load_model_data(model_file, model_key):
    """Load data for a specific model"""
    try:
        with open(model_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract relevant data
        records = []
        for article in data:
            # Get source rating
            source_rating = None
            if 'source_rating_value' in article:
                source_rating = article['source_rating_value']
            elif 'source_rating_value_precise' in article:
                source_rating = article['source_rating_value_precise']
            
            # Convert to float
            try:
                source_rating = float(source_rating)
            except (ValueError, TypeError):
                source_rating = None
                
            # Get AI rating
            ai_rating = None
            if model_key == 'deepseek' and 'ai_political_rating' in article:
                ai_rating = article['ai_political_rating']
            elif model_key == 'openai' and 'openai_political_rating' in article:
                ai_rating = article['openai_political_rating']
            elif model_key == 'gemini' and 'gemini_political_rating' in article:
                ai_rating = article['gemini_political_rating']
            
            # Convert to float
            try:
                ai_rating = float(ai_rating)
            except (ValueError, TypeError):
                ai_rating = None
                
            # Only include valid entries
            if source_rating is not None and ai_rating is not None:
                source_category = get_category_from_rating(source_rating)
                records.append({
                    'source_rating': source_rating,
                    'ai_rating': ai_rating,
                    'source_category': source_category,
                    'error': ai_rating - source_rating  # Calculate error
                })
        
        return pd.DataFrame(records)
    
    except Exception as e:
        print(f"Error loading {model_key} data: {e}")
        return pd.DataFrame()

def create_error_distribution_plot():
    """Create an error distribution histogram comparing all models"""
    # Set global font parameters for publication quality
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.weight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold'
    })
    
    # Load data for each model
    all_data = {}
    error_data = {}
    mae_values = {}
    
    for config in MODEL_CONFIGS:
        model_key = config['key']
        df = load_model_data(config['file'], model_key)
        
        if not df.empty:
            all_data[model_key] = df
            error_data[model_key] = df['error'].values
            mae_values[model_key] = mean_absolute_error(df['source_rating'], df['ai_rating'])
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create histogram bins (match the example with appropriate range and bins)
    bins = np.linspace(-6, 6, 25)  # 25 bins gives us a good distribution
    bin_width = bins[1] - bins[0]
    
    # Add background shading for error categories
    ax.axvspan(-6, -2, alpha=0.15, color='red', label='Large Error (>2)')
    ax.axvspan(-2, -1, alpha=0.15, color='orange', label='Medium Error (1-2)')
    ax.axvspan(-1, 1, alpha=0.15, color='green', label='Small Error (<1)')
    ax.axvspan(1, 2, alpha=0.15, color='orange')
    ax.axvspan(2, 6, alpha=0.15, color='red')
    
    # Calculate number of models for bar positioning
    active_models = [config for config in MODEL_CONFIGS if config['key'] in error_data]
    num_models = len(active_models)
    
    # Calculate bar width based on number of models
    model_bar_width = bin_width / (num_models + 0.2)  # Leave a small gap between bins
    
    # Plot side-by-side (dodged) bars for each model
    for i, config in enumerate(active_models):
        model_key = config['key']
        # Calculate histogram data
        hist_counts, hist_bins = np.histogram(error_data[model_key], bins=bins)
        # Calculate bar positions (center of each bin with offset for each model)
        bar_positions = [(hist_bins[j] + hist_bins[j+1])/2 - bin_width/2 + (i + 0.5) * model_bar_width 
                        for j in range(len(hist_bins)-1)]
        
        # Plot bars for this model
        ax.bar(bar_positions, hist_counts, width=model_bar_width, alpha=0.85,
              label=f"{config['name']}", color=config['color'],
              edgecolor='black', linewidth=1.0)
    
    # Add perfect agreement line - make it more prominent
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2.5, label='Perfect Agreement')
    
    # Set axis labels and title
    ax.set_xlabel('Error (AI Rating - AllSides Rating)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Articles', fontsize=16, fontweight='bold')
    
    # Set axis limits to match the data
    ax.set_xlim(-6, 6)
    
    # Add grid for readability
    ax.grid(True, alpha=0.3)
    
    # Create combined legend for both shading and models
    # First create custom patch legends for the error categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.15, label='Small Error (<1)', edgecolor='black', linewidth=0.5),
        Patch(facecolor='orange', alpha=0.15, label='Medium Error (1-2)', edgecolor='black', linewidth=0.5),
        Patch(facecolor='red', alpha=0.15, label='Large Error (>2)', edgecolor='black', linewidth=0.5)
    ]
    
    # Add model entries to legend elements
    for config in MODEL_CONFIGS:
        model_key = config['key']
        if model_key in error_data:
            legend_elements.append(Patch(facecolor=config['color'], alpha=0.65, 
                                        label=f"{config['name']}",
                                        edgecolor='black', linewidth=1.0))
    
    # Add perfect agreement line to legend
    from matplotlib.lines import Line2D
    legend_elements.insert(0, Line2D([0], [0], color='black', linestyle='--', linewidth=2.5,
                                    label='Perfect Agreement'))
    
    # Add specific tick marks for clarity
    ax.set_xticks(range(-6, 7))  # Show every integer from -6 to 6
    
    # Add more tick marks on the y-axis
    y_max = ax.get_ylim()[1]
    ax.set_yticks(np.arange(0, y_max + 1, 5))  # Add y-axis ticks every 5 units
    
    # Adjust layout
    plt.tight_layout()
    
    # Create and position the legend with a border - smaller size
    ax.legend(handles=legend_elements, loc='upper right', fontsize=16, 
             framealpha=0.9, edgecolor='black', fancybox=True, 
             prop={'size': 16, 'weight': 'bold'},
             borderpad=1.2, labelspacing=1.2, handlelength=3.5, handletextpad=1.0)
    
    # Add a border around the plot for clarity
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, "error_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved error distribution plot to {output_path}")
    
    return True

def main():
    """Main function"""
    create_error_distribution_plot()

if __name__ == "__main__":
    main() 