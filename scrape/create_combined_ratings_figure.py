#!/usr/bin/env python3
"""
Create a combined figure showing AI model ratings vs source ratings for all three models.
This script generates a readable visualization that combines all three models in one figure.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.lines import Line2D

# Create figures directory if it doesn't exist
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures/combined")
os.makedirs(FIGURES_DIR, exist_ok=True)

# File paths for the AI model results
DEEPSEEK_FILE = 'balanced_dataset/cleaned_articles_rated_deepseek.json'
OPENAI_FILE = 'balanced_dataset/cleaned_articles_rated_openai.json'
GEMINI_FILE = 'balanced_dataset/cleaned_articles_rated_gemini.json'

# Define category ranges and colors
CATEGORY_RANGES = {
    "Left": (-6, -3),
    "Lean Left": (-3, -1),
    "Center": (-1, 1),
    "Lean Right": (1, 3),
    "Right": (3, 6)
}

CATEGORY_COLORS = {
    "Left": "#3333FF",       # Blue
    "Lean Left": "#99CCFF",  # Light blue
    "Center": "#AAAAAA",     # Grey
    "Lean Right": "#FFCC99", # Light red
    "Right": "#FF3333"       # Red
}

# Model colors
MODEL_COLORS = {
    'deepseek': '#1f77b4',  # blue
    'openai': '#ff7f0e',    # orange
    'gemini': '#2ca02c'     # green
}

# Model name mapping
MODEL_NAMES = {
    'deepseek': 'DeepSeek V3',
    'openai': 'OpenAI GPT-4o',
    'gemini': 'Google Gemini Flash 2.0'
}

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

def load_and_merge_data():
    """Load and merge data from all models"""
    all_data = {}
    
    # Load data from each model
    for model, file in [('deepseek', DEEPSEEK_FILE), ('openai', OPENAI_FILE), ('gemini', GEMINI_FILE)]:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
                print(f"Loaded {len(model_data)} articles from {model}")
                all_data[model] = model_data
        except Exception as e:
            print(f"Error loading {model} data: {e}")
    
    # Check if we have data for all models
    if len(all_data) < 3:
        print("Missing data for one or more models. Please check the input files.")
        return None
    
    # Create a merged dataframe with all models
    df = pd.DataFrame()
    
    # Get unique articles by link
    all_links = set()
    for model_data in all_data.values():
        all_links.update([article.get('link', '') for article in model_data])
    
    # Create records for each article
    records = []
    for link in all_links:
        record = {'link': link}
        
        # Find this article in each model's data
        for model, model_data in all_data.items():
            article = next((a for a in model_data if a.get('link', '') == link), None)
            if article:
                # Add basic metadata (use the first available data)
                if 'title' not in record and 'title' in article:
                    record['title'] = article['title']
                if 'source_outlet' not in record and 'source_outlet' in article:
                    record['source_outlet'] = article['source_outlet']
                if 'source_rating' not in record and 'source_rating' in article:
                    record['source_rating'] = article['source_rating']
                    
                # Try different field names for source rating value
                if 'source_rating_value' not in record:
                    source_rating_value = None
                    if 'source_rating_value' in article:
                        source_rating_value = article['source_rating_value']
                    elif 'source_rating_value_precise' in article:
                        source_rating_value = article['source_rating_value_precise']
                        
                    # Convert to float if possible
                    try:
                        if source_rating_value is not None:
                            source_rating_value = float(source_rating_value)
                    except (ValueError, TypeError):
                        source_rating_value = None
                        
                    record['source_rating_value'] = source_rating_value
                
                # Add model-specific ratings
                ai_rating = None
                if model == 'deepseek' and 'ai_political_rating' in article:
                    ai_rating = article['ai_political_rating']
                elif model == 'openai' and 'openai_political_rating' in article:
                    ai_rating = article['openai_political_rating']
                elif model == 'gemini' and 'gemini_political_rating' in article:
                    ai_rating = article['gemini_political_rating']
                    
                # Convert to float if possible
                try:
                    if ai_rating is not None:
                        ai_rating = float(ai_rating)
                except (ValueError, TypeError):
                    ai_rating = None
                    
                record[f'{model}_rating'] = ai_rating
        
        # Only include records with source ratings and at least one model rating
        if ('source_rating_value' in record and record['source_rating_value'] is not None and
            any(f'{model}_rating' in record and record[f'{model}_rating'] is not None 
                for model in all_data.keys())):
            records.append(record)
    
    # Create dataframe
    df = pd.DataFrame(records)
    
    # Ensure all numerical columns are float
    numeric_columns = ['source_rating_value'] + [f'{model}_rating' for model in all_data.keys()]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"Created dataframe with {len(df)} articles")
    return df

def create_combined_figure(df):
    """Create a combined figure with all three models vs source ratings"""
    models = ['deepseek', 'openai', 'gemini']
    
    # Verify we have all models in the data
    available_models = [model for model in models if f'{model}_rating' in df.columns]
    if len(available_models) < len(models):
        print(f"Warning: Missing data for models: {set(models) - set(available_models)}")
    
    # Calculate correlations for each model
    correlations = {}
    for model in available_models:
        model_df = df.dropna(subset=[f'{model}_rating', 'source_rating_value'])
        corr, _ = pearsonr(model_df['source_rating_value'], model_df[f'{model}_rating'])
        correlations[model] = corr
    
    # Create figure with subplots (2 rows, 2 columns)
    fig = plt.figure(figsize=(15, 15))
    
    # 1. Top left: Combined regression lines (no scatter points)
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    
    # Plot regression lines for each model
    for model in available_models:
        model_df = df.dropna(subset=[f'{model}_rating', 'source_rating_value'])
        sns.regplot(x='source_rating_value', y=f'{model}_rating', data=model_df, 
                   scatter=False, 
                   color=MODEL_COLORS[model],
                   line_kws={'label': f"{MODEL_NAMES[model]} (r={correlations[model]:.2f})",
                            'linewidth': 3})
    
    # Add diagonal line (perfect agreement)
    ax1.plot([-6, 6], [-6, 6], 'k--', alpha=0.7, linewidth=2, label="Perfect Agreement")
    
    # Add vertical and horizontal lines at category boundaries
    for boundary in [-3, -1, 1, 3]:
        ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
        ax1.axhline(y=boundary, color='gray', linestyle='--', alpha=0.3)
    
    # Add category labels
    for category, (min_val, max_val) in CATEGORY_RANGES.items():
        mid = (min_val + max_val) / 2
        # X-axis labels
        ax1.text(mid, -6.5, category, ha='center', fontsize=10)
        # Y-axis labels
        ax1.text(-6.5, mid, category, va='center', fontsize=10, rotation=90)
    
    ax1.set_title("AI Models vs Source Ratings - Regression Lines", fontsize=18, fontweight='bold')
    ax1.set_xlabel("Source Rating (-6 to 6)", fontsize=14)
    ax1.set_ylabel("AI Rating (-6 to 6)", fontsize=14)
    ax1.set_xlim(-6.5, 6.5)
    ax1.set_ylim(-6.5, 6.5)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=12, loc='upper left')
    
    # 2-4. Individual scatter plots for each model (bottom row)
    for i, model in enumerate(available_models):
        ax = plt.subplot2grid((2, 2), (1, i % 2))
        
        model_df = df.dropna(subset=[f'{model}_rating', 'source_rating_value'])
        
        # Create scatter plot
        ax.scatter(model_df['source_rating_value'], model_df[f'{model}_rating'], 
                 color=MODEL_COLORS[model], alpha=0.6, s=50)
        
        # Add regression line
        m, b = np.polyfit(model_df['source_rating_value'], model_df[f'{model}_rating'], 1)
        ax.plot([-6, 6], [m*-6+b, m*6+b], color=MODEL_COLORS[model], linewidth=2)
        
        # Add diagonal line (perfect agreement)
        ax.plot([-6, 6], [-6, 6], 'k--', alpha=0.5)
        
        # Add vertical and horizontal lines at category boundaries
        for boundary in [-3, -1, 1, 3]:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.2)
            ax.axhline(y=boundary, color='gray', linestyle='--', alpha=0.2)
        
        # Add statistics to plot
        stats_text = f"Correlation: {correlations[model]:.2f}\nSample Size: {len(model_df)}"
        ax.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                   ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # Set title and labels
        ax.set_title(f"{MODEL_NAMES[model]} vs Source Ratings", fontsize=14)
        ax.set_xlabel("Source Rating (-6 to 6)", fontsize=12)
        ax.set_ylabel("AI Rating (-6 to 6)", fontsize=12)
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-6.5, 6.5)
        ax.grid(alpha=0.2)
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = os.path.join(FIGURES_DIR, "combined_ai_vs_source_ratings.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved combined figure to {output_file}")
    
    # Create alternative layout with just the regression lines (simpler)
    plt.figure(figsize=(12, 9))
    
    # Plot regression lines for each model
    for model in available_models:
        model_df = df.dropna(subset=[f'{model}_rating', 'source_rating_value'])
        sns.regplot(x='source_rating_value', y=f'{model}_rating', data=model_df, 
                   scatter=True, 
                   scatter_kws={'alpha': 0.3, 's': 20, 'color': MODEL_COLORS[model]},
                   color=MODEL_COLORS[model],
                   line_kws={'label': f"{MODEL_NAMES[model]} (r={correlations[model]:.2f})",
                            'linewidth': 3})
    
    # Add diagonal line (perfect agreement)
    plt.plot([-6, 6], [-6, 6], 'k--', alpha=0.7, linewidth=2, label="Perfect Agreement")
    
    # Add vertical and horizontal lines at category boundaries
    for boundary in [-3, -1, 1, 3]:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=boundary, color='gray', linestyle='--', alpha=0.3)
    
    # Add category labels
    for category, (min_val, max_val) in CATEGORY_RANGES.items():
        # Create colored areas in the background
        plt.axvspan(min_val, max_val, alpha=0.05, color=CATEGORY_COLORS[category])
        
        # Add category label on x-axis
        plt.text((min_val + max_val) / 2, -6.5, category, ha='center', fontsize=12)
    
    plt.title("AI Models vs Source Ratings", fontsize=20, fontweight='bold')
    plt.xlabel("Source Rating (-6 to 6)", fontsize=16)
    plt.ylabel("AI Rating (-6 to 6)", fontsize=16)
    plt.xlim(-6.5, 6.5)
    plt.ylim(-6.5, 6.5)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14, loc='upper left')
    
    # Create custom legend for the categories
    category_handles = [Line2D([0], [0], marker='s', color='w', 
                              markerfacecolor=CATEGORY_COLORS[cat], markersize=15, label=cat)
                      for cat in CATEGORY_RANGES.keys()]
    
    # Add second legend for categories
    second_legend = plt.legend(handles=category_handles, loc='lower right', 
                              title='Source Categories', fontsize=12)
    
    # Add the first legend back
    plt.gca().add_artist(plt.gca().get_legend())
    plt.gca().add_artist(second_legend)
    
    # Save the alternative figure
    alt_output_file = os.path.join(FIGURES_DIR, "combined_ai_vs_source_simple.png")
    plt.savefig(alt_output_file, dpi=300, bbox_inches='tight')
    print(f"Saved alternative combined figure to {alt_output_file}")

def main():
    """Main function"""
    # Load and merge data
    df = load_and_merge_data()
    if df is not None:
        # Create combined figure
        create_combined_figure(df)

if __name__ == "__main__":
    main() 