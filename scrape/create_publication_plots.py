#!/usr/bin/env python3
"""
Create publication-quality plots for comparing AI models with source ratings.
Generates a side-by-side comparison of all three models in one figure.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# Create figures directory if it doesn't exist
OUTPUT_DIR = "figures/publication"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# File paths for the AI model results
DEEPSEEK_FILE = 'balanced_dataset/cleaned_articles_rated_deepseek.json'
OPENAI_FILE = 'balanced_dataset/cleaned_articles_rated_openai.json'
GEMINI_FILE = 'balanced_dataset/cleaned_articles_rated_gemini.json'

# Define category ranges and colors for political bias
CATEGORY_RANGES = {
    "Left": (-6, -3),
    "Lean Left": (-3, -1),
    "Center": (-1, 1),
    "Lean Right": (1, 3),
    "Right": (3, 6)
}

# Colors matching the provided example
CATEGORY_COLORS = {
    "Left": "#3333FF",       # Blue
    "Lean Left": "#99CCFF",  # Light blue
    "Center": "#AAAAAA",     # Grey
    "Lean Right": "#FFcc99", # Light orange
    "Right": "#FF3333"       # Red
}

# Model configurations
MODEL_CONFIGS = [
    {'name': 'OpenAI GPT-4o', 'file': OPENAI_FILE, 'key': 'openai', 'subplot_label': '(a)'},
    {'name': 'DeepSeek V3', 'file': DEEPSEEK_FILE, 'key': 'deepseek', 'subplot_label': '(b)'},
    {'name': 'Google Gemini Flash 2.0', 'file': GEMINI_FILE, 'key': 'gemini', 'subplot_label': '(c)'}
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
                    'source_category': source_category
                })
        
        return pd.DataFrame(records)
    
    except Exception as e:
        print(f"Error loading {model_key} data: {e}")
        return pd.DataFrame()

def calculate_metrics(df):
    """Calculate performance metrics for the model"""
    # Calculate correlation
    correlation, _ = pearsonr(df['source_rating'], df['ai_rating'])
    
    # Calculate MAE
    mae = mean_absolute_error(df['source_rating'], df['ai_rating'])
    
    # Calculate category accuracy
    df['ai_category'] = df['ai_rating'].apply(get_category_from_rating)
    category_accuracy = (df['ai_category'] == df['source_category']).mean() * 100
    
    # Calculate directional accuracy
    direction_match = (
        (df['source_rating'] > 0) & (df['ai_rating'] > 0) |
        (df['source_rating'] < 0) & (df['ai_rating'] < 0) |
        (df['source_rating'] == 0) & (abs(df['ai_rating']) <= 1)
    )
    directional_accuracy = direction_match.mean() * 100
    
    return {
        'correlation': correlation,
        'mae': mae,
        'category_accuracy': category_accuracy,
        'directional_accuracy': directional_accuracy,
        'sample_size': len(df)
    }

def create_publication_plots():
    """Create publication-quality plots for all models"""
    # Load data for each model
    all_data = {}
    for config in MODEL_CONFIGS:
        model_key = config['key']
        all_data[model_key] = load_model_data(config['file'], model_key)
    
    # Calculate metrics for each model
    metrics = {}
    for model_key, df in all_data.items():
        if not df.empty:
            metrics[model_key] = calculate_metrics(df)
    
    # Create figure with subplots - increase bottom margin for legend
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True, sharex=True)
    fig.subplots_adjust(bottom=0.25)  # Increased from 0.2 to 0.25 to make more room
    
    # Create plots for each model
    for i, config in enumerate(MODEL_CONFIGS):
        model_key = config['key']
        ax = axes[i]
        
        if model_key in all_data and not all_data[model_key].empty:
            df = all_data[model_key]
            model_metrics = metrics[model_key]
            
            # Create scatter plot with color based on source category
            scatter = ax.scatter(
                df['source_rating'], 
                df['ai_rating'],
                c=[CATEGORY_COLORS.get(cat, "#000000") for cat in df['source_category']],
                alpha=0.7, 
                s=40
            )
            
            # Add diagonal perfect agreement line
            ax.plot([-6, 6], [-6, 6], 'k--', alpha=0.6)
            
            # Add green band for "close enough" agreement
            ax.fill_between([-6, 6], [-6-1, 6-1], [-6+1, 6+1], color='green', alpha=0.05)
            
            # Add grid lines at category boundaries
            for boundary in [-3, -1, 1, 3]:
                ax.axvline(x=boundary, color='lightgray', linestyle='--', alpha=0.6)
                ax.axhline(y=boundary, color='lightgray', linestyle='--', alpha=0.6)
            
            # Add metrics text
            metrics_text = (
                f"r = {model_metrics['correlation']:.2f}\n"
                f"MAE = {model_metrics['mae']:.2f}\n"
                f"Cat Acc = {model_metrics['category_accuracy']:.1f}%\n"
                f"Dir Acc = {model_metrics['directional_accuracy']:.1f}%\n"
                f"n = {model_metrics['sample_size']}"
            )
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                   va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
            
            # Add subplot label (a, b, c)
            ax.text(0.95, 0.95, config['subplot_label'], transform=ax.transAxes,
                   va='top', ha='right', fontsize=14, fontweight='bold')
            
            # Set title
            ax.set_title(config['name'], fontsize=14, fontweight='bold')
            
            # Set axis labels for leftmost plot and bottom plots
            if i == 0:
                ax.set_ylabel('AI Rating (-6 to 6)', fontsize=12)
            ax.set_xlabel('AllSides Rating (-6 to 6)', fontsize=12)
            
            # Set axis limits
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
    
    # Create a single legend for the entire figure - position further down
    legend_elements = []
    for category, color in CATEGORY_COLORS.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                        markersize=10, label=category))
    
    # Position legend lower by setting bbox_to_anchor y value to -0.1 (below the figure)
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
              title='Source Category', bbox_to_anchor=(0.5, 0), fontsize=10)
    
    # Adjust layout with wider padding at bottom for legend
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # rect=[left, bottom, right, top]
    
    output_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved publication figure to {output_path}")
    
    return True

def main():
    """Main function"""
    create_publication_plots()

if __name__ == "__main__":
    main() 