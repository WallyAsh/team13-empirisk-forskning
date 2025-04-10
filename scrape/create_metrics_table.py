#!/usr/bin/env python3
"""
Create a performance metrics table comparing DeepSeek, OpenAI GPT-4o, and Gemini models.
This script generates a nicely formatted table similar to the requested format.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error

# File paths for the AI model results
DEEPSEEK_FILE = 'balanced_dataset/cleaned_articles_rated_deepseek.json'
OPENAI_FILE = 'balanced_dataset/cleaned_articles_rated_openai.json'
GEMINI_FILE = 'balanced_dataset/cleaned_articles_rated_gemini.json'

# Define category ranges for political bias
CATEGORY_RANGES = {
    "Left": (-6, -3),
    "Lean Left": (-3, -1),
    "Center": (-1, 1),
    "Lean Right": (1, 3),
    "Right": (3, 6)
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

        # Calculate directional match (same sign)
        df[f'{model}_directional_match'] = (
            (df['source_rating_value'] > 0) & (df[f'ai_{model}_rating'] > 0) |
            (df['source_rating_value'] < 0) & (df[f'ai_{model}_rating'] < 0) |
            (df['source_rating_value'] == 0) & (abs(df[f'ai_{model}_rating']) <= 1)
        )
    
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
        directional_accuracy = model_df[f'{model}_directional_match'].mean() * 100
        
        # Count articles by error magnitude
        small_error = (model_df[f'{model}_abs_error'] < 1).sum()
        medium_error = ((model_df[f'{model}_abs_error'] >= 1) & (model_df[f'{model}_abs_error'] < 2)).sum()
        large_error = (model_df[f'{model}_abs_error'] >= 2).sum()
        
        # Store metrics
        metrics[model] = {
            'correlation': correlation,
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

def create_formatted_metrics_table(metrics):
    """Create a formatted metrics table in the requested layout"""
    if not metrics:
        print("No metrics available to create table")
        return
    
    # Define metrics in order
    metrics_order = [
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
    
    # Create data for each model
    model_data = {}
    model_names = {
        'deepseek': 'DeepSeek V3',
        'openai': 'GPT-4o',
        'gemini': 'Gemini Flash 2.0'
    }
    
    for model, model_metrics in metrics.items():
        model_data[model_names.get(model, model.capitalize())] = [
            f"{model_metrics.get('correlation', 0):.3f}",
            f"{model_metrics.get('r_squared', 0):.3f}",
            f"{model_metrics.get('mae', 0):.3f}",
            f"{model_metrics.get('rmse', 0):.3f}",
            f"{model_metrics.get('category_accuracy', 0):.1f}%",
            f"{model_metrics.get('directional_accuracy', 0):.1f}%",
            f"{model_metrics.get('small_error_percent', 0):.1f}%",
            f"{model_metrics.get('medium_error_percent', 0):.1f}%",
            f"{model_metrics.get('large_error_percent', 0):.1f}%",
            f"{model_metrics.get('sample_size', 0)}"
        ]
    
    # Create DataFrame
    df = pd.DataFrame(model_data, index=metrics_order)
    
    # Save as CSV with metrics as the first column
    df.reset_index(names='Metric').to_csv('performance_metrics.csv', index=False)
    
    # Format for display
    print("\nPerformance Metrics Table:\n")
    print(df.to_string())
    
    # Print information about saved file
    print("\nTable saved to performance_metrics.csv")
    
    return df

def main():
    """Main function to create metrics table"""
    # Load all AI model data
    all_data = load_data()
    if not all_data:
        return
    
    # Merge data into a single DataFrame
    df = merge_data(all_data)
    
    # Calculate metrics for each model
    metrics = calculate_model_metrics(df)
    
    # Create and print formatted metrics table
    metrics_table = create_formatted_metrics_table(metrics)

if __name__ == "__main__":
    main() 