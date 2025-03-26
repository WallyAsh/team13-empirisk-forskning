#!/usr/bin/env python3
"""
Update the error distribution plot with larger text and better readability.
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
        
        # Calculate error metrics
        mae = mean_absolute_error(model_df['source_rating_value'], model_df[f'ai_{model}_rating'])
        
        # Store metrics
        metrics[model] = {
            'mae': mae,
            'sample_size': len(model_df)
        }
    
    return metrics

def create_error_distribution_plot(df, metrics):
    """Create a more readable error distribution plot"""
    plt.figure(figsize=(16, 10))
    
    # Set global font sizes for better readability
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })
    
    models = ['deepseek', 'openai', 'gemini']
    model_names = {'deepseek': 'DeepSeek', 'openai': 'OpenAI GPT-4o', 'gemini': 'Google Gemini'}
    colors = {'deepseek': '#1f77b4', 'openai': '#ff7f0e', 'gemini': '#2ca02c'}
    
    for model in models:
        if f'{model}_error' not in df.columns:
            continue
            
        # Filter out NaNs
        model_df = df.dropna(subset=[f'{model}_error'])
        
        if len(model_df) == 0:
            continue
        
        # Plot error distribution with more prominent elements
        sns.histplot(model_df[f'{model}_error'], bins=20, 
                  alpha=0.7, 
                  label=f"{model_names.get(model, model.capitalize())} (MAE={metrics.get(model, {}).get('mae', 0):.2f})",
                  color=colors.get(model, None),
                  linewidth=2,
                  edgecolor='black')
    
    # Add perfect agreement line with increased visibility
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.8, linewidth=2.5, label="Perfect Agreement")
    
    # Add shaded regions for error categories with increased contrast
    plt.axvspan(-1, 1, alpha=0.15, color='green', label="Small Error (<1)")
    plt.axvspan(-2, -1, alpha=0.15, color='yellow')
    plt.axvspan(1, 2, alpha=0.15, color='yellow', label="Medium Error (1-2)")
    plt.axvspan(-6, -2, alpha=0.15, color='red')
    plt.axvspan(2, 6, alpha=0.15, color='red', label="Large Error (>2)")
    
    # Add title and labels with increased font size
    plt.title("Error Distribution by AI Model", fontsize=24, fontweight='bold', pad=15)
    plt.xlabel("Error (AI Rating - AllSides Rating)", fontsize=20, fontweight='bold')
    plt.ylabel("Count", fontsize=20, fontweight='bold')
    
    # Increase legend size and visibility
    plt.legend(fontsize=18, frameon=True, framealpha=0.95, edgecolor='black')
    
    # Improved grid
    plt.grid(alpha=0.3, linewidth=1.0)
    
    # Make the plot border thicker
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2.0)
    
    # Add more prominent tick marks
    plt.tick_params(width=2.0, length=8, labelsize=16)
    
    # Save the plot with higher DPI
    output_file = os.path.join(FIGURES_DIR, "error_distribution_improved.png")
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    print(f"Improved error distribution plot saved to {output_file}")
    
    return output_file

def main():
    """Main function to update the error distribution plot"""
    # Load all AI model data
    all_data = load_data()
    if not all_data:
        return
    
    # Merge data into a single DataFrame
    df = merge_data(all_data)
    
    # Calculate metrics for each model
    metrics = calculate_model_metrics(df)
    
    # Create improved error distribution plot
    output_file = create_error_distribution_plot(df, metrics)
    
    print(f"\nImproved error distribution plot created: {output_file}")
    print("You can use this file in your LaTeX document.")

if __name__ == "__main__":
    main() 