#!/usr/bin/env python3
"""
Compare political bias ratings from multiple AI models with source ratings for the balanced dataset.

This script:
1. Loads article ratings from all three AI models (DeepSeek, OpenAI, and Gemini)
2. Compares AI ratings with source ratings in the balanced dataset
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

# Create figures directory if it doesn't exist
FIGURES_DIR = "figures/balanced_dataset"
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
    
    # Remove articles with no source ratings
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
        corr, p_value = pearsonr(model_df[f'ai_{model}_rating'], model_df['source_rating_value'])
        
        # Calculate MAE and RMSE
        mae = mean_absolute_error(model_df['source_rating_value'], model_df[f'ai_{model}_rating'])
        rmse = np.sqrt(mean_squared_error(model_df['source_rating_value'], model_df[f'ai_{model}_rating']))
        
        # Calculate category match percentage
        category_match_pct = model_df[f'{model}_category_match'].mean() * 100
        
        # Calculate bias (mean error)
        mean_error = model_df[f'{model}_error'].mean()
        
        # Calculate RMSE per category
        rmse_by_category = {}
        for category in CATEGORY_RANGES.keys():
            category_df = model_df[model_df['source_rating'] == category]
            if len(category_df) > 0:
                cat_rmse = np.sqrt(mean_squared_error(
                    category_df['source_rating_value'], 
                    category_df[f'ai_{model}_rating']
                ))
                rmse_by_category[category] = cat_rmse
        
        # Store metrics
        metrics[model] = {
            'correlation': corr,
            'p_value': p_value,
            'mae': mae,
            'rmse': rmse,
            'category_match_pct': category_match_pct,
            'bias': mean_error,
            'rmse_by_category': rmse_by_category,
            'n_samples': len(model_df)
        }
        
        print(f"\n{model.upper()} Metrics:")
        print(f"  Correlation: {corr:.3f} (p={p_value:.6f})")
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  Category Match: {category_match_pct:.1f}%")
        print(f"  Bias (mean error): {mean_error:.3f}")
        print(f"  Number of samples: {len(model_df)}")
    
    return metrics

def create_comparison_scatterplot(df, metrics):
    """Create a scatterplot comparing AI ratings with AllSides ratings"""
    models = ['deepseek', 'openai', 'gemini']
    
    for model in models:
        # Skip if no data for this model
        if f'ai_{model}_rating' not in df.columns or df[f'ai_{model}_rating'].notna().sum() == 0:
            continue
            
        # Filter out NaN values
        model_df = df.dropna(subset=[f'ai_{model}_rating', 'source_rating_value'])
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Set up the plot
        sns.set_style("whitegrid")
        
        # Define colors based on source rating category
        colors = [CATEGORY_COLORS[get_category_from_rating(val)] for val in model_df['source_rating_value']]
        
        # Create scatterplot
        plt.scatter(
            model_df['source_rating_value'],
            model_df[f'ai_{model}_rating'],
            c=colors,
            alpha=0.7,
            s=50,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add labels and title
        plt.xlabel("Source Rating (AllSides)", fontsize=14)
        plt.ylabel(f"{model.capitalize()} AI Rating", fontsize=14)
        
        model_name_map = {
            'deepseek': 'DeepSeek',
            'openai': 'GPT-4o',
            'gemini': 'Gemini'
        }
        
        model_name = model_name_map.get(model, model.capitalize())
        
        model_metrics = metrics.get(model, {})
        corr = model_metrics.get('correlation', float('nan'))
        rmse = model_metrics.get('rmse', float('nan'))
        category_match = model_metrics.get('category_match_pct', float('nan'))
        
        plt.title(
            f"{model_name} vs. Source Ratings\n"
            f"Correlation: {corr:.3f}, RMSE: {rmse:.3f}, Category Match: {category_match:.1f}%",
            fontsize=16
        )
        
        # Add diagonal line (perfect correlation)
        plt.plot([-6, 6], [-6, 6], 'k--', alpha=0.5, linewidth=1)
        
        # Add horizontal and vertical lines for category boundaries
        for boundary in [-3, -1, 1, 3]:
            plt.axvline(x=boundary, color='gray', alpha=0.3, linestyle='--')
            plt.axhline(y=boundary, color='gray', alpha=0.3, linestyle='--')
        
        # Set axis limits
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        
        # Add colorbar legend for categories
        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=category)
            for category, color in CATEGORY_COLORS.items()
        ]
        plt.legend(handles=handles, title="Source Rating Category", loc='upper left')
        
        # Add category labels to both axes
        category_positions = {
            "Left": -4.5,
            "Lean Left": -2,
            "Center": 0,
            "Lean Right": 2,
            "Right": 4.5
        }
        
        # X-axis category labels
        for category, pos in category_positions.items():
            plt.text(
                pos, -6.2, category,
                ha='center', va='top',
                fontsize=10, color=CATEGORY_COLORS[category],
                fontweight='bold'
            )
        
        # Y-axis category labels
        for category, pos in category_positions.items():
            plt.text(
                -6.2, pos, category,
                ha='right', va='center',
                fontsize=10, color=CATEGORY_COLORS[category],
                rotation=90, fontweight='bold'
            )
        
        # Set tight layout and save
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        filename = os.path.join(FIGURES_DIR, f"{model}_vs_source_ratings.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        
        plt.close()

def create_model_comparison_plot(df):
    """Create comparative boxplots showing error distributions for each model by category"""
    # Prepare data in long format for seaborn
    plot_data = []
    
    models = ['deepseek', 'openai', 'gemini']
    model_names = {'deepseek': 'DeepSeek', 'openai': 'GPT-4o', 'gemini': 'Gemini'}
    
    # Collect error data by category for each model
    for category in CATEGORY_RANGES.keys():
        category_df = df[df['source_rating'] == category]
        
        for model in models:
            model_errors = category_df[f'{model}_error'].dropna()
            
            # Skip if no data
            if len(model_errors) == 0:
                continue
                
            for error in model_errors:
                plot_data.append({
                    'Model': model_names.get(model, model.capitalize()),
                    'Source Category': category,
                    'Error': error
                })
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Set up the plot
    sns.set_style("whitegrid")
    
    # Create box plot
    ax = sns.boxplot(
        x='Source Category',
        y='Error',
        hue='Model',
        data=plot_df,
        palette=['#1f77b4', '#ff7f0e', '#2ca02c'],  # Blue, Orange, Green
        fliersize=3
    )
    
    # Add horizontal line at y=0 (no error)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel("Source Rating Category", fontsize=14)
    plt.ylabel("Error (AI Rating - Source Rating)", fontsize=14)
    plt.title("Error Distribution by Category and Model", fontsize=16)
    
    # Adjust legend
    plt.legend(title="AI Model", fontsize=12)
    
    # Set tight layout and save
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    filename = os.path.join(FIGURES_DIR, "model_error_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    
    plt.close()

def create_error_distribution_plot(df, metrics):
    """Create a KDE plot showing error distribution for each model"""
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set up the plot
    sns.set_style("whitegrid")
    
    # Define models and colors
    models = ['deepseek', 'openai', 'gemini']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    linestyles = ['-', '--', '-.']
    
    # Plot KDE for each model
    for i, model in enumerate(models):
        # Skip if no data for this model
        if f'{model}_error' not in df.columns or df[f'{model}_error'].notna().sum() == 0:
            continue
            
        model_errors = df[f'{model}_error'].dropna()
        
        # Get metrics for this model
        model_metrics = metrics.get(model, {})
        bias = model_metrics.get('bias', float('nan'))
        rmse = model_metrics.get('rmse', float('nan'))
        
        # Plot KDE
        sns.kdeplot(
            model_errors,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            label=f"{model.capitalize()}: Mean={bias:.2f}, RMSE={rmse:.2f}",
            linewidth=3
        )
    
    # Add vertical line at x=0 (no error)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel("Error (AI Rating - Source Rating)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("Error Distribution by Model", fontsize=16)
    
    # Adjust legend
    plt.legend(title="AI Model", fontsize=12)
    
    # Set tight layout and save
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    filename = os.path.join(FIGURES_DIR, "error_distribution.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {filename}")
    
    plt.close()

def create_metrics_table(metrics):
    """Create a table showing performance metrics for each model"""
    if not metrics:
        print("No metrics available to create table")
        return
    
    # Create DataFrame for table
    rows = []
    for model, model_metrics in metrics.items():
        row = {
            'Model': model.capitalize(),
            'Correlation': model_metrics.get('correlation', float('nan')),
            'MAE': model_metrics.get('mae', float('nan')),
            'RMSE': model_metrics.get('rmse', float('nan')),
            'Category Match (%)': model_metrics.get('category_match_pct', float('nan')),
            'Bias': model_metrics.get('bias', float('nan')),
            'N': model_metrics.get('n_samples', 0)
        }
        
        # Add RMSE by category
        rmse_by_cat = model_metrics.get('rmse_by_category', {})
        for category in CATEGORY_RANGES.keys():
            row[f'RMSE_{category}'] = rmse_by_cat.get(category, float('nan'))
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Format numeric columns
    format_dict = {
        'Correlation': '{:.3f}',
        'MAE': '{:.3f}',
        'RMSE': '{:.3f}',
        'Category Match (%)': '{:.1f}%',
        'Bias': '{:.3f}'
    }
    
    for category in CATEGORY_RANGES.keys():
        format_dict[f'RMSE_{category}'] = '{:.3f}'
    
    # Apply formatting
    for col, fmt in format_dict.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt.format(x) if not pd.isna(x) else 'N/A')
    
    # Save as CSV
    filename = os.path.join(FIGURES_DIR, "model_metrics.csv")
    df.to_csv(filename, index=False)
    print(f"Saved metrics table: {filename}")
    
    # Print table
    print("\nModel Performance Metrics:")
    print(df.to_string(index=False))

def create_confusion_matrices(df):
    """Create confusion matrices for each model's category predictions"""
    models = ['deepseek', 'openai', 'gemini']
    categories = list(CATEGORY_RANGES.keys())
    
    for model in models:
        # Skip if no data for this model
        if f'ai_{model}_category_derived' not in df.columns or df[f'ai_{model}_category_derived'].notna().sum() == 0:
            continue
            
        # Get source and AI categories
        model_df = df.dropna(subset=[f'ai_{model}_category_derived', 'source_rating_value'])
        
        true_categories = model_df['source_rating'].values
        pred_categories = model_df[f'ai_{model}_category_derived'].values
        
        # Create confusion matrix
        cm = confusion_matrix(
            [categories.index(cat) for cat in true_categories],
            [categories.index(cat) if cat in categories else -1 for cat in pred_categories],
            labels=range(len(categories))
        )
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=categories,
            yticklabels=categories,
            linewidths=.5,
            linecolor='gray'
        )
        
        # Add labels and title
        plt.xlabel('Predicted Category', fontsize=14)
        plt.ylabel('True Category', fontsize=14)
        plt.title(f'{model.capitalize()} Category Predictions', fontsize=16)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Set tight layout and save
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(FIGURES_DIR, f"{model}_confusion_matrix.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
        
        plt.close()

def main():
    """Main function to run the analysis"""
    print("Starting AI model comparison analysis...")
    
    # Load data from all models
    all_data = load_data()
    if not all_data:
        print("No data available. Exiting.")
        return
    
    # Merge data for analysis
    df = merge_data(all_data)
    if df.empty:
        print("No valid data after merging. Exiting.")
        return
        
    # Calculate metrics
    metrics = calculate_model_metrics(df)
    
    # Create visualizations
    create_comparison_scatterplot(df, metrics)
    create_model_comparison_plot(df)
    create_error_distribution_plot(df, metrics)
    create_metrics_table(metrics)
    create_confusion_matrices(df)
    
    print("\nAnalysis complete. All figures saved to:", FIGURES_DIR)

if __name__ == "__main__":
    main() 