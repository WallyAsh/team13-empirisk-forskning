#!/usr/bin/env python3
"""
Compare political bias ratings from multiple AI models with AllSides ratings.

This script:
1. Loads article ratings from all three AI models (DeepSeek, OpenAI, and Gemini)
2. Compares AI ratings with AllSides source ratings
3. Generates statistics and visualizations comparing model performance
4. Saves figures for scientific reporting
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from adjustText import adjust_text

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
    """Load and merge AI ratings data from all models"""
    all_data = {}
    
    # Load DeepSeek data
    try:
        if os.path.exists(DEEPSEEK_FILE):
            with open(DEEPSEEK_FILE, 'r', encoding='utf-8') as f:
                deepseek_data = json.load(f)
                print(f"Loaded {len(deepseek_data)} articles from DeepSeek")
                all_data['deepseek'] = deepseek_data
    except Exception as e:
        print(f"Error loading DeepSeek data: {e}")
    
    # Load OpenAI data
    try:
        if os.path.exists(OPENAI_FILE):
            with open(OPENAI_FILE, 'r', encoding='utf-8') as f:
                openai_data = json.load(f)
                print(f"Loaded {len(openai_data)} articles from OpenAI")
                all_data['openai'] = openai_data
    except Exception as e:
        print(f"Error loading OpenAI data: {e}")
    
    # Load Gemini data
    try:
        if os.path.exists(GEMINI_FILE):
            with open(GEMINI_FILE, 'r', encoding='utf-8') as f:
                gemini_data = json.load(f)
                print(f"Loaded {len(gemini_data)} articles from Gemini")
                all_data['gemini'] = gemini_data
    except Exception as e:
        print(f"Error loading Gemini data: {e}")
    
    # Check if we have at least one model's data
    if not all_data:
        print("No AI model data found. Please run the classification scripts first.")
        return None
    
    return all_data

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
    if 'deepseek' in all_data:
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
    if 'openai' in all_data:
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
    if 'gemini' in all_data:
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
    
    # Remove articles with no AI ratings or source ratings
    df = df.dropna(subset=['source_rating_value'])
    
    # Print information about the data
    print(f"\nMerged {len(df)} articles with valid source ratings")
    print(f"Articles with DeepSeek ratings: {df['ai_deepseek_rating'].notna().sum()}")
    print(f"Articles with OpenAI ratings: {df['ai_openai_rating'].notna().sum()}")
    print(f"Articles with Gemini ratings: {df['ai_gemini_rating'].notna().sum()}")
    
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

def create_comparison_scatterplot(df, metrics):
    """Create scatterplots comparing each AI model to AllSides ratings"""
    models = ['deepseek', 'openai', 'gemini']
    model_names = {'deepseek': 'DeepSeek', 'openai': 'OpenAI GPT-4o', 'gemini': 'Google Gemini'}
    
    # Create comparison plot for each model
    for model in models:
        if f'ai_{model}_rating' not in df.columns:
            continue
            
        # Filter data for this model
        model_df = df.dropna(subset=[f'ai_{model}_rating', 'source_rating_value'])
        
        if len(model_df) == 0:
            print(f"No valid data for {model} model scatterplot")
            continue
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        # Color points by source category
        colors = [CATEGORY_COLORS.get(cat, "#000000") for cat in model_df['source_rating']]
        
        # Create scatter plot
        plt.scatter(model_df['source_rating_value'], model_df[f'ai_{model}_rating'], 
                  c=colors, alpha=0.7, s=50)
        
        # Add diagonal line (perfect agreement)
        plt.plot([-6, 6], [-6, 6], 'k--', alpha=0.5, label="Perfect Agreement")
        
        # Add vertical and horizontal lines at category boundaries
        for boundary in [-3, -1, 1, 3]:
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
            plt.axhline(y=boundary, color='gray', linestyle='--', alpha=0.3)
        
        # Add category labels
        for category, (min_val, max_val) in CATEGORY_RANGES.items():
            # X-axis (source) labels
            plt.text((min_val + max_val) / 2, -6.5, category, ha='center', fontsize=8)
            # Y-axis (AI) labels
            plt.text(-6.5, (min_val + max_val) / 2, category, va='center', fontsize=8, rotation=90)
        
        # Group by source outlet and calculate statistics
        outlet_stats = model_df.groupby('source_outlet').agg({
            f'{model}_error': 'mean',
            f'{model}_abs_error': 'mean',
            f'ai_{model}_rating': ['mean', 'count'],
            'source_rating_value': 'mean',
            'source_rating': 'first'
        }).reset_index()
        
        # Flatten the column structure
        outlet_stats.columns = ['source_outlet', 'mean_error', 'mean_abs_error', 'mean_ai', 'count', 'mean_source', 'source_category']
        
        # Filter to outlets with at least 3 articles
        major_outlets = outlet_stats[outlet_stats['count'] >= 3].sort_values('count', ascending=False)
        
        # Label major outlets on the plot
        texts = []
        for i, row in major_outlets.iterrows():
            # Only label outlets with significant count or disagreement
            if row['count'] >= 3 or abs(row['mean_error']) > 1.5:
                txt = plt.text(row['mean_source'], row['mean_ai'], 
                         row['source_outlet'], 
                         fontsize=9, fontweight='bold')
                texts.append(txt)
        
        # Adjust text to avoid overlap
        if texts:
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
        
        # Get metrics for this model
        model_metrics = metrics.get(model, {})
        
        # Add statistics to plot
        stats_text = (
            f"Correlation: {model_metrics.get('correlation', 0):.2f} (R² = {model_metrics.get('r_squared', 0):.2f})\n"
            f"Mean Abs Error: {model_metrics.get('mae', 0):.2f}\n"
            f"RMSE: {model_metrics.get('rmse', 0):.2f}\n"
            f"Same Category: {model_metrics.get('category_accuracy', 0):.1f}%\n"
            f"Same Direction: {model_metrics.get('directional_accuracy', 0):.1f}%\n"
            f"Small Error (<1): {model_metrics.get('small_error_percent', 0):.1f}%\n"
            f"Medium Error (1-2): {model_metrics.get('medium_error_percent', 0):.1f}%\n"
            f"Large Error (>2): {model_metrics.get('large_error_percent', 0):.1f}%\n"
            f"Sample Size: {model_metrics.get('sample_size', 0)}"
        )
        
        plt.annotate(stats_text, xy=(0.03, 0.97), xycoords='axes fraction',
                    ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # Create legend for source categories
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color, markersize=10, label=category)
                        for category, color in CATEGORY_COLORS.items()]
        
        # Add diagonal band for "close enough" agreement
        plt.fill_between([-6, 6], [-7, 5], [-5, 7], color='green', alpha=0.05)
        
        # Add legend
        plt.legend(handles=legend_elements, title="Source Category", loc='lower right')
        
        # Set plot title and labels
        plt.title(f"{model_names.get(model, model.capitalize())} Rating vs AllSides Rating", fontsize=16)
        plt.xlabel("AllSides Precise Rating (-6 to 6)", fontsize=14)
        plt.ylabel(f"{model_names.get(model, model.capitalize())} Rating (-6 to 6)", fontsize=14)
        
        # Set axis limits
        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        
        # Add grid
        plt.grid(alpha=0.2)
        
        # Save the plot
        output_file = os.path.join(FIGURES_DIR, f"{model}_vs_allsides.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Comparison plot for {model} saved to {output_file}")
    
    # Create regression plot for all models
    plt.figure(figsize=(15, 10))
    
    for model in models:
        if f'ai_{model}_rating' not in df.columns:
            continue
            
        # Filter data for this model
        model_df = df.dropna(subset=[f'ai_{model}_rating', 'source_rating_value'])
        
        if len(model_df) == 0:
            continue
        
        # Plot regression line
        sns.regplot(x='source_rating_value', y=f'ai_{model}_rating', data=model_df, 
                    scatter=False, 
                    line_kws={'label': f"{model_names.get(model, model.capitalize())} (r={metrics.get(model, {}).get('correlation', 0):.2f})"})
    
    # Add diagonal line (perfect agreement)
    plt.plot([-6, 6], [-6, 6], 'k--', alpha=0.5, label="Perfect Agreement")
    
    # Add vertical and horizontal lines at category boundaries
    for boundary in [-3, -1, 1, 3]:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=boundary, color='gray', linestyle='--', alpha=0.3)
    
    plt.title("AI Models vs AllSides Ratings Regression Lines", fontsize=16)
    plt.xlabel("AllSides Precise Rating (-6 to 6)", fontsize=14)
    plt.ylabel("AI Political Rating (-6 to 6)", fontsize=14)
    plt.xlim(-6.5, 6.5)
    plt.ylim(-6.5, 6.5)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=12)
    
    # Save the combined regression plot
    regression_file = os.path.join(FIGURES_DIR, "all_models_regression.png")
    plt.savefig(regression_file, dpi=300, bbox_inches='tight')
    print(f"Combined regression plot saved to {regression_file}")

def create_model_comparison_plot(df):
    """Create plots comparing the three AI models directly"""
    models = ['deepseek', 'openai', 'gemini']
    model_names = {'deepseek': 'DeepSeek', 'openai': 'OpenAI GPT-4o', 'gemini': 'Google Gemini'}
    
    # Create pairwise comparison plots
    model_pairs = [('deepseek', 'openai'), ('deepseek', 'gemini'), ('openai', 'gemini')]
    
    for model1, model2 in model_pairs:
        # Filter for articles that have both models' ratings
        pair_df = df.dropna(subset=[f'ai_{model1}_rating', f'ai_{model2}_rating'])
        
        if len(pair_df) == 0:
            print(f"No common articles between {model1} and {model2}")
            continue
        
        # Calculate correlation between models
        correlation, p_value = pearsonr(pair_df[f'ai_{model1}_rating'], pair_df[f'ai_{model2}_rating'])
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot colored by source category
        scatter = plt.scatter(pair_df[f'ai_{model1}_rating'], pair_df[f'ai_{model2}_rating'], 
                            c=[CATEGORY_COLORS.get(cat, "#000000") for cat in pair_df['source_rating']], 
                            alpha=0.7, s=60)
        
        # Add diagonal line (perfect agreement)
        plt.plot([-6, 6], [-6, 6], 'k--', alpha=0.5)
        
        # Add vertical and horizontal lines at category boundaries
        for boundary in [-3, -1, 1, 3]:
            plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
            plt.axhline(y=boundary, color='gray', linestyle='--', alpha=0.3)
        
        # Add category labels and regions
        for category, (min_val, max_val) in CATEGORY_RANGES.items():
            # Add colored rectangles for category regions
            plt.fill_between([min_val, max_val], [min_val, max_val], [max_val, max_val], 
                          alpha=0.05, color=CATEGORY_COLORS.get(category, '#CCCCCC'))
            plt.fill_between([min_val, max_val], [min_val, min_val], [min_val, max_val], 
                          alpha=0.05, color=CATEGORY_COLORS.get(category, '#CCCCCC'))
        
        # Create legend for source categories
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color, markersize=10, label=category)
                        for category, color in CATEGORY_COLORS.items()]
        
        # Calculate category agreement percentage
        category_agreement = (pair_df[f'ai_{model1}_category_derived'] == pair_df[f'ai_{model2}_category_derived']).mean() * 100
        
        # Calculate root mean squared difference
        rmsd = np.sqrt(mean_squared_error(pair_df[f'ai_{model1}_rating'], pair_df[f'ai_{model2}_rating']))
        
        # Add statistics to plot
        stats_text = (
            f"Correlation: {correlation:.2f} (R² = {correlation**2:.2f})\n"
            f"Category Agreement: {category_agreement:.1f}%\n"
            f"RMSD: {rmsd:.2f}\n"
            f"Sample Size: {len(pair_df)}"
        )
        
        plt.annotate(stats_text, xy=(0.03, 0.97), xycoords='axes fraction',
                    ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # Add legend
        plt.legend(handles=legend_elements, title="Source Category", loc='lower right')
        
        # Set plot title and labels
        plt.title(f"{model_names.get(model1, model1.capitalize())} vs {model_names.get(model2, model2.capitalize())} Ratings", fontsize=16)
        plt.xlabel(f"{model_names.get(model1, model1.capitalize())} Rating (-6 to 6)", fontsize=14)
        plt.ylabel(f"{model_names.get(model2, model2.capitalize())} Rating (-6 to 6)", fontsize=14)
        
        # Set axis limits
        plt.xlim(-6.5, 6.5)
        plt.ylim(-6.5, 6.5)
        
        # Add grid
        plt.grid(alpha=0.2)
        
        # Save the plot
        output_file = os.path.join(FIGURES_DIR, f"{model1}_vs_{model2}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Model comparison plot saved to {output_file}")

def create_error_distribution_plot(df, metrics):
    """Create histogram of error distributions for all models"""
    plt.figure(figsize=(15, 10))
    
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
        
        # Plot error distribution
        sns.histplot(model_df[f'{model}_error'], bins=20, 
                 alpha=0.6, 
                 label=f"{model_names.get(model, model.capitalize())} (MAE={metrics.get(model, {}).get('mae', 0):.2f})",
                 color=colors.get(model, None))
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, label="Perfect Agreement")
    
    # Add shaded regions for error categories
    plt.axvspan(-1, 1, alpha=0.1, color='green', label="Small Error (<1)")
    plt.axvspan(-2, -1, alpha=0.1, color='yellow')
    plt.axvspan(1, 2, alpha=0.1, color='yellow', label="Medium Error (1-2)")
    plt.axvspan(-6, -2, alpha=0.1, color='red')
    plt.axvspan(2, 6, alpha=0.1, color='red', label="Large Error (>2)")
    
    plt.title("Error Distribution by AI Model", fontsize=16)
    plt.xlabel("Error (AI Rating - AllSides Rating)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    
    # Save the plot
    output_file = os.path.join(FIGURES_DIR, "error_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Error distribution plot saved to {output_file}")

def create_metrics_table(metrics):
    """Create a comparative metrics table for the report"""
    if not metrics:
        return
    
    models = list(metrics.keys())
    model_names = {'deepseek': 'DeepSeek', 'openai': 'OpenAI GPT-4o', 'gemini': 'Google Gemini'}
    
    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame({
        'Metric': [
            'Correlation (r)', 
            'R-squared', 
            'Mean Absolute Error',
            'Root Mean Squared Error',
            'Category Accuracy (%)',
            'Directional Accuracy (%)',
            'Small Error (<1) (%)',
            'Medium Error (1-2) (%)',
            'Large Error (>2) (%)',
            'Sample Size'
        ]
    })
    
    # Add columns for each model
    for model in models:
        if model in metrics:
            metrics_df[model_names.get(model, model.capitalize())] = [
                f"{metrics[model].get('correlation', 0):.3f}",
                f"{metrics[model].get('r_squared', 0):.3f}",
                f"{metrics[model].get('mae', 0):.3f}",
                f"{metrics[model].get('rmse', 0):.3f}",
                f"{metrics[model].get('category_accuracy', 0):.1f}%",
                f"{metrics[model].get('directional_accuracy', 0):.1f}%",
                f"{metrics[model].get('small_error_percent', 0):.1f}%",
                f"{metrics[model].get('medium_error_percent', 0):.1f}%",
                f"{metrics[model].get('large_error_percent', 0):.1f}%",
                f"{metrics[model].get('sample_size', 0)}"
            ]
    
    # Save the metrics as CSV
    output_file = os.path.join(FIGURES_DIR, "model_metrics.csv")
    metrics_df.to_csv(output_file, index=False)
    print(f"Metrics table saved to {output_file}")
    
    # Format and print the table for console output
    print("\n===== AI MODEL PERFORMANCE COMPARISON =====")
    print(metrics_df.to_string(index=False))
    print("===========================================\n")
    
    return metrics_df

def create_confusion_matrices(df):
    """Create confusion matrices for categorical predictions"""
    models = ['deepseek', 'openai', 'gemini']
    model_names = {'deepseek': 'DeepSeek', 'openai': 'OpenAI GPT-4o', 'gemini': 'Google Gemini'}
    categories = ["Left", "Lean Left", "Center", "Lean Right", "Right"]
    
    for model in models:
        if f'ai_{model}_category_derived' not in df.columns:
            continue
            
        # Filter valid data
        model_df = df.dropna(subset=[f'ai_{model}_category_derived', 'source_rating'])
        
        if len(model_df) == 0:
            continue
        
        # Map source ratings to derived categories for consistent comparison
        source_categories = []
        for source_rating in model_df['source_rating']:
            # Find the source_rating_value for this source_rating
            source_value = model_df.loc[model_df['source_rating'] == source_rating, 'source_rating_value'].iloc[0]
            # Get category from rating
            source_categories.append(get_category_from_rating(source_value))
        
        # Get AI categories
        ai_categories = model_df[f'ai_{model}_category_derived'].tolist()
        
        # Compute confusion matrix
        cm = confusion_matrix(source_categories, ai_categories, labels=categories)
        
        # Create normalized confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot confusion matrix
        sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                 xticklabels=categories, yticklabels=categories,
                 annot_kws={"size": 14}, cbar=False)
        
        plt.title(f"{model_names.get(model, model.capitalize())} Classification Confusion Matrix", fontsize=16)
        plt.xlabel('AI Predicted Category', fontsize=14)
        plt.ylabel('AllSides Category', fontsize=14)
        
        # Add accuracy in the plot
        accuracy = np.trace(cm) / np.sum(cm) * 100
        plt.text(0.5, -0.1, f"Accuracy: {accuracy:.1f}%", 
              horizontalalignment='center',
              fontsize=14,
              transform=plt.gca().transAxes)
        
        # Save the plot
        output_file = os.path.join(FIGURES_DIR, f"{model}_confusion_matrix.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix for {model} saved to {output_file}")

def main():
    """Main function to run the entire analysis and create visualizations"""
    # Load all AI model data
    all_data = load_data()
    if not all_data:
        return
    
    # Merge data into a single DataFrame
    df = merge_data(all_data)
    
    # Calculate metrics for each model
    metrics = calculate_model_metrics(df)
    
    # Create comparison scatter plots
    create_comparison_scatterplot(df, metrics)
    
    # Create model comparison plots
    create_model_comparison_plot(df)
    
    # Create error distribution plots
    create_error_distribution_plot(df, metrics)
    
    # Create confusion matrices
    create_confusion_matrices(df)
    
    # Create and print metrics table
    metrics_df = create_metrics_table(metrics)
    
    print("\nAnalysis complete! All figures saved to the 'figures' directory.")
    print("These visualizations can be used in a scientific report to compare the performance of the three AI models.")

if __name__ == "__main__":
    main() 