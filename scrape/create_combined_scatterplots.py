#!/usr/bin/env python3
"""
Create a combined figure with all three AI vs AllSides scatter plots in subplots.
Each subplot will be labeled (a), (b), (c) for publication in LaTeX.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create figures directory if it doesn't exist
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# File paths for the AI model results
DEEPSEEK_FILE = 'best_articles/top5_per_source_rated_deepseek.json'
OPENAI_FILE = 'best_articles/top5_per_source_rated_openai.json'
GEMINI_FILE = 'best_articles/top5_per_source_rated_gemini.json'

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

# Function to get category from numerical rating
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

def load_data():
    """Load AI ratings data from all models"""
    data = {'deepseek': None, 'openai': None, 'gemini': None}
    
    try:
        with open(DEEPSEEK_FILE, 'r', encoding='utf-8') as f:
            data['deepseek'] = json.load(f)
            print(f"Loaded {len(data['deepseek'])} articles from DeepSeek")
    except Exception as e:
        print(f"Error loading DeepSeek data: {e}")
    
    try:
        with open(OPENAI_FILE, 'r', encoding='utf-8') as f:
            data['openai'] = json.load(f)
            print(f"Loaded {len(data['openai'])} articles from OpenAI")
    except Exception as e:
        print(f"Error loading OpenAI data: {e}")
    
    try:
        with open(GEMINI_FILE, 'r', encoding='utf-8') as f:
            data['gemini'] = json.load(f)
            print(f"Loaded {len(data['gemini'])} articles from Gemini")
    except Exception as e:
        print(f"Error loading Gemini data: {e}")
    
    return data

def merge_data(all_data):
    """Merge AI ratings from different models into a single DataFrame for analysis"""
    # Initialize an empty list to store merged data
    merged_records = []
    
    # Start with DeepSeek data if available
    base_model = next(iter(all_data.values()))
    
    for article in base_model:
        # Try different field names for source rating value
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
            
        record = {
            # Article metadata
            'title': article.get('title', ''),
            'link': article.get('link', ''),
            'pub_date': article.get('pub_date', ''),
            'source_outlet': article.get('source_outlet', ''),
            'source_rating': article.get('source_rating', ''),
            'source_rating_value': source_rating_value,
            
            # Placeholder for AI ratings
            'ai_deepseek_rating': None,
            'ai_deepseek_category': None,
            'ai_openai_rating': None,
            'ai_openai_category': None,
            'ai_gemini_rating': None,
            'ai_gemini_category': None
        }
        
        # Add to merged data
        merged_records.append(record)
    
    # Create a DataFrame
    df = pd.DataFrame(merged_records)
    
    # Create a dictionary to easily find articles by link
    articles_by_link = {article['link']: i for i, article in enumerate(merged_records)}
    
    # Update with DeepSeek ratings if available
    if 'deepseek' in all_data and all_data['deepseek']:
        for article in all_data['deepseek']:
            link = article.get('link', '')
            if link in articles_by_link:
                idx = articles_by_link[link]
                rating = article.get('ai_political_rating')
                # Convert to float if possible
                try:
                    if rating is not None:
                        rating = float(rating)
                except (ValueError, TypeError):
                    rating = None
                df.at[idx, 'ai_deepseek_rating'] = rating
                df.at[idx, 'ai_deepseek_category'] = article.get('ai_political_leaning')
    
    # Update with OpenAI ratings if available
    if 'openai' in all_data and all_data['openai']:
        for article in all_data['openai']:
            link = article.get('link', '')
            if link in articles_by_link:
                idx = articles_by_link[link]
                rating = article.get('openai_political_rating')
                # Convert to float if possible
                try:
                    if rating is not None:
                        rating = float(rating)
                except (ValueError, TypeError):
                    rating = None
                df.at[idx, 'ai_openai_rating'] = rating
                df.at[idx, 'ai_openai_category'] = article.get('openai_political_leaning')
    
    # Update with Gemini ratings if available
    if 'gemini' in all_data and all_data['gemini']:
        for article in all_data['gemini']:
            link = article.get('link', '')
            if link in articles_by_link:
                idx = articles_by_link[link]
                rating = article.get('gemini_political_rating')
                # Convert to float if possible
                try:
                    if rating is not None:
                        rating = float(rating)
                except (ValueError, TypeError):
                    rating = None
                df.at[idx, 'ai_gemini_rating'] = rating
                df.at[idx, 'ai_gemini_category'] = article.get('gemini_political_leaning')
    
    # Ensure all numerical columns are float
    numeric_columns = ['source_rating_value', 'ai_deepseek_rating', 'ai_openai_rating', 'ai_gemini_rating']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove articles with no source ratings
    df = df.dropna(subset=['source_rating_value'])
    
    # Add derived columns for analysis
    for model in ['deepseek', 'openai', 'gemini']:
        # Skip if no articles have this model's ratings
        if df[f'ai_{model}_rating'].notna().sum() == 0:
            continue
            
        # Calculate error metrics
        df[f'{model}_error'] = df[f'ai_{model}_rating'] - df['source_rating_value']
        df[f'{model}_abs_error'] = abs(df[f'{model}_error'])
        
        # Ensure categories match the numerical ratings
        df[f'ai_{model}_category_derived'] = df[f'ai_{model}_rating'].apply(get_category_from_rating)
        
        # Calculate category match
        source_categories = df['source_rating_value'].apply(get_category_from_rating)
        df[f'{model}_category_match'] = df[f'ai_{model}_category_derived'] == source_categories
    
    return df

def calculate_model_metrics(df):
    """Calculate performance metrics for each AI model"""
    metrics = {}
    models = ['deepseek', 'openai', 'gemini']
    
    for model in models:
        # Filter out NaN values for this model
        model_df = df.dropna(subset=[f'ai_{model}_rating', 'source_rating_value'])
        
        if len(model_df) == 0:
            print(f"No valid data for {model} model")
            continue
        
        # Calculate correlation with source ratings
        correlation, p_value = pearsonr(model_df['source_rating_value'], model_df[f'ai_{model}_rating'])
        r_squared = correlation**2
        
        # Calculate error metrics
        mae = mean_absolute_error(model_df['source_rating_value'], model_df[f'ai_{model}_rating'])
        rmse = np.sqrt(mean_squared_error(model_df['source_rating_value'], model_df[f'ai_{model}_rating']))
        
        # Calculate category accuracy
        category_accuracy = model_df[f'{model}_category_match'].mean() * 100
        
        # Calculate directional accuracy (same sign)
        directional_match = (
            (model_df['source_rating_value'] > 0) & (model_df[f'ai_{model}_rating'] > 0) |
            (model_df['source_rating_value'] < 0) & (model_df[f'ai_{model}_rating'] < 0) |
            (model_df['source_rating_value'] == 0) & (abs(model_df[f'ai_{model}_rating']) <= 1)
        )
        directional_accuracy = directional_match.mean() * 100
        
        # Count articles by error magnitude
        small_error = (model_df[f'{model}_abs_error'] < 1).sum()
        medium_error = ((model_df[f'{model}_abs_error'] >= 1) & (model_df[f'{model}_abs_error'] < 2)).sum()
        large_error = (model_df[f'{model}_abs_error'] >= 2).sum()
        
        # Store metrics
        metrics[model] = {
            'correlation': correlation,
            'p_value': p_value,
            'r_squared': r_squared,
            'mae': mae,
            'rmse': rmse,
            'category_accuracy': category_accuracy,
            'directional_accuracy': directional_accuracy,
            'small_error_count': small_error,
            'small_error_percent': small_error / len(model_df) * 100,
            'medium_error_count': medium_error,
            'medium_error_percent': medium_error / len(model_df) * 100,
            'large_error_count': large_error,
            'large_error_percent': large_error / len(model_df) * 100,
            'sample_size': len(model_df)
        }
    
    return metrics

def create_combined_scatterplots(df, metrics):
    """Create a combined figure with all three AI vs AllSides plots in subplots"""
    # Define models for each subplot in the order (a), (b), (c)
    model_order = ['openai', 'deepseek', 'gemini']
    model_names = {'deepseek': 'DeepSeek R1', 'openai': 'OpenAI GPT-4o', 'gemini': 'Google Gemini Flash 2.0'}
    subplot_labels = ['(a)', '(b)', '(c)']
    
    # Set global font sizes for better readability
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16,  # Increased legend font size
    })
    
    # Create a figure with three subplots in a row with adjusted size
    fig, axes = plt.subplots(1, 3, figsize=(24, 10), sharey=True, sharex=True)
    
    # Create each subplot
    for i, model in enumerate(model_order):
        ax = axes[i]
        
        # Filter data for this model
        model_df = df.dropna(subset=[f'ai_{model}_rating', 'source_rating_value'])
        
        if len(model_df) == 0:
            print(f"No valid data for {model} model scatterplot")
            continue
        
        # Color points by source category
        colors = [CATEGORY_COLORS.get(cat, "#000000") for cat in model_df['source_rating']]
        
        # Create scatter plot on the appropriate subplot with larger points
        ax.scatter(model_df['source_rating_value'], model_df[f'ai_{model}_rating'], 
                 c=colors, alpha=0.8, s=70, edgecolors='black', linewidths=0.5)
        
        # Add diagonal line (perfect agreement)
        ax.plot([-6, 6], [-6, 6], 'k--', alpha=0.6, linewidth=2, label="Perfect Agreement")
        
        # Add vertical and horizontal lines at category boundaries
        for boundary in [-3, -1, 1, 3]:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
            ax.axhline(y=boundary, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
        
        # Add diagonal band for "close enough" agreement
        ax.fill_between([-6, 6], [-7, 5], [-5, 7], color='green', alpha=0.05)
        
        # Get metrics for this model
        model_metrics = metrics.get(model, {})
        
        # Add statistics to plot with larger font and more visible box
        stats_text = (
            f"r = {model_metrics.get('correlation', 0):.2f}\n"
            f"MAE = {model_metrics.get('mae', 0):.2f}\n"
            f"Cat Acc = {model_metrics.get('category_accuracy', 0):.1f}%\n"
            f"Dir Acc = {model_metrics.get('directional_accuracy', 0):.1f}%\n"
            f"n = {model_metrics.get('sample_size', 0)}"
        )
        
        ax.annotate(stats_text, xy=(0.03, 0.97), xycoords='axes fraction',
                   ha='left', va='top', fontsize=16,
                   bbox=dict(boxstyle='round', fc='white', alpha=0.85, edgecolor='black', pad=0.5))
        
        # Add subplot label (a), (b), or (c) with larger font
        ax.annotate(subplot_labels[i], xy=(0.95, 0.95), xycoords='axes fraction', 
                   fontsize=22, fontweight='bold', ha='right', va='top')
        
        # Set plot title for each subplot
        ax.set_title(f"{model_names.get(model, model.capitalize())}", fontsize=18, fontweight='bold', pad=10)
        
        # Set x and y labels only for the first subplot (to avoid repetition)
        if i == 0:
            ax.set_ylabel("AI Rating (-6 to 6)", fontsize=16, fontweight='bold')
        
        ax.set_xlabel("AllSides Rating (-6 to 6)", fontsize=16, fontweight='bold')
        
        # Set axis limits
        ax.set_xlim(-6.5, 6.5)
        ax.set_ylim(-6.5, 6.5)
        
        # Add grid with better visibility
        ax.grid(alpha=0.25)
        
        # Thicker spines for better visibility
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # Add more prominent tick marks
        ax.tick_params(width=1.5, length=6)
    
    # Create a common legend for all subplots with larger markers
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=color, markersize=20,  # Increased marker size
                           markeredgecolor='black', markeredgewidth=0.5,
                           label=category)
                   for category, color in CATEGORY_COLORS.items()]
    
    # Use the figure to create a single legend for all subplots
    # Moved legend closer to plots by adjusting bbox_to_anchor
    fig.legend(handles=legend_elements, title="Source Category", title_fontsize=18,  # Increased title size
              loc='lower center', bbox_to_anchor=(0.5, -0.03), ncol=5, fontsize=16,  # Adjusted position and size
              frameon=True, framealpha=0.95, edgecolor='black')
    
    # Adjust layout
    plt.tight_layout()
    # Reduced bottom margin to bring legend closer to plots
    plt.subplots_adjust(bottom=0.13)
    
    # Save the combined figure with higher DPI
    output_file = os.path.join(FIGURES_DIR, "combined_scatterplots.png")
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    print(f"Combined scatter plots saved to {output_file}")
    
    return output_file

def main():
    """Main function to create the combined scatter plot figure"""
    # Load all AI model data
    all_data = load_data()
    if not all_data:
        return
    
    # Merge data into a single DataFrame
    df = merge_data(all_data)
    
    # Calculate metrics for each model
    metrics = calculate_model_metrics(df)
    
    # Create combined scatter plots
    output_file = create_combined_scatterplots(df, metrics)
    
    print(f"\nCombined figure created: {output_file}")
    print("You can use this file in your LaTeX document as shown in your template.")

if __name__ == "__main__":
    main() 