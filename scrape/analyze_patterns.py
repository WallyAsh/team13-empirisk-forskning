#!/usr/bin/env python3
"""
Analyze patterns in AI model ratings and errors.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Create figures directory if it doesn't exist
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# File paths for the AI model results
DEEPSEEK_FILE = 'best_articles/top5_per_source_rated_deepseek.json'
OPENAI_FILE = 'best_articles/top5_per_source_rated_openai.json'
GEMINI_FILE = 'best_articles/top5_per_source_rated_gemini.json'

def load_and_merge_data():
    """Load and merge data from all models"""
    # Load data from each model
    data = {'deepseek': None, 'openai': None, 'gemini': None}
    
    for model, file in [('deepseek', DEEPSEEK_FILE), ('openai', OPENAI_FILE), ('gemini', GEMINI_FILE)]:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data[model] = json.load(f)
                print(f"Loaded {len(data[model])} articles from {model}")
        except Exception as e:
            print(f"Error loading {model} data: {e}")
    
    # Create DataFrame
    records = []
    for model, articles in data.items():
        if not articles:
            continue
            
        for article in articles:
            source_rating = None
            if 'source_rating_value' in article:
                source_rating = article['source_rating_value']
            elif 'source_rating_value_precise' in article:
                source_rating = article['source_rating_value_precise']
                
            try:
                source_rating = float(source_rating)
            except (ValueError, TypeError):
                source_rating = None
                
            ai_rating = None
            if model == 'deepseek':
                ai_rating = article.get('ai_political_rating')
            elif model == 'openai':
                ai_rating = article.get('openai_political_rating')
            elif model == 'gemini':
                ai_rating = article.get('gemini_political_rating')
                
            try:
                ai_rating = float(ai_rating)
            except (ValueError, TypeError):
                ai_rating = None
                
            records.append({
                'model': model,
                'title': article.get('title', ''),
                'source_outlet': article.get('source_outlet', ''),
                'source_rating': source_rating,
                'ai_rating': ai_rating,
                'error': ai_rating - source_rating if (ai_rating is not None and source_rating is not None) else None
            })
    
    df = pd.DataFrame(records)
    return df

def analyze_bias_by_outlet(df):
    """Analyze patterns in errors by news outlet"""
    # Calculate mean error and error variance by outlet
    outlet_stats = df.groupby(['source_outlet', 'model']).agg({
        'error': ['mean', 'std', 'count']
    }).reset_index()
    
    outlet_stats.columns = ['source_outlet', 'model', 'mean_error', 'std_error', 'count']
    
    # Sort by absolute mean error
    outlet_stats['abs_mean_error'] = abs(outlet_stats['mean_error'])
    outlet_stats = outlet_stats.sort_values('abs_mean_error', ascending=False)
    
    print("\nTop outlets with largest average errors:")
    print(outlet_stats.head(10).to_string(index=False))
    
    # Create error by outlet plot
    plt.figure(figsize=(15, 8))
    
    # Get top 10 outlets by absolute mean error
    top_outlets = outlet_stats.groupby('source_outlet')['abs_mean_error'].mean().nlargest(10).index
    
    # Filter data for plot
    plot_data = outlet_stats[outlet_stats['source_outlet'].isin(top_outlets)]
    
    # Create grouped bar plot
    sns.barplot(data=plot_data, x='source_outlet', y='mean_error', hue='model')
    
    plt.title('Mean Error by News Outlet (Top 10 by Absolute Error)', fontsize=14, pad=20)
    plt.xlabel('News Outlet', fontsize=12)
    plt.ylabel('Mean Error (AI - AllSides)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='AI Model')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'error_by_outlet.png'))
    print(f"\nError by outlet plot saved to {os.path.join(FIGURES_DIR, 'error_by_outlet.png')}")

def analyze_error_patterns(df):
    """Analyze patterns in rating errors"""
    # Calculate error statistics by source rating
    error_stats = df.groupby(['source_rating', 'model']).agg({
        'error': ['mean', 'std', 'count']
    }).reset_index()
    
    error_stats.columns = ['source_rating', 'model', 'mean_error', 'std_error', 'count']
    
    print("\nError patterns by source rating:")
    print(error_stats.to_string(index=False))
    
    # Create error pattern plot
    plt.figure(figsize=(12, 8))
    
    for model in df['model'].unique():
        model_data = error_stats[error_stats['model'] == model]
        plt.plot(model_data['source_rating'], model_data['mean_error'], 
                marker='o', label=model, linewidth=2)
        
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title('Mean Error by Source Rating', fontsize=14, pad=20)
    plt.xlabel('AllSides Rating', fontsize=12)
    plt.ylabel('Mean Error (AI - AllSides)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(title='AI Model')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'error_by_rating.png'))
    print(f"\nError by rating plot saved to {os.path.join(FIGURES_DIR, 'error_by_rating.png')}")

def analyze_agreement_patterns(df):
    """Analyze patterns in model agreement"""
    # Create pivot table for model comparisons
    pivot_df = df.pivot(index=['title', 'source_outlet', 'source_rating'], 
                       columns='model', 
                       values='ai_rating').reset_index()
    
    # Calculate agreement metrics
    agreement_stats = {
        'deepseek_vs_openai': stats.pearsonr(
            pivot_df['deepseek'].dropna(), 
            pivot_df['openai'].dropna()
        )[0],
        'deepseek_vs_gemini': stats.pearsonr(
            pivot_df['deepseek'].dropna(), 
            pivot_df['gemini'].dropna()
        )[0],
        'openai_vs_gemini': stats.pearsonr(
            pivot_df['openai'].dropna(), 
            pivot_df['gemini'].dropna()
        )[0]
    }
    
    print("\nModel Agreement Correlations:")
    for comparison, corr in agreement_stats.items():
        print(f"{comparison}: r = {corr:.3f}")
    
    # Create agreement heatmap
    plt.figure(figsize=(8, 6))
    
    # Create correlation matrix
    corr_matrix = np.array([
        [1.0, agreement_stats['deepseek_vs_openai'], agreement_stats['deepseek_vs_gemini']],
        [agreement_stats['deepseek_vs_openai'], 1.0, agreement_stats['openai_vs_gemini']],
        [agreement_stats['deepseek_vs_gemini'], agreement_stats['openai_vs_gemini'], 1.0]
    ])
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu', 
                xticklabels=['DeepSeek', 'OpenAI', 'Gemini'],
                yticklabels=['DeepSeek', 'OpenAI', 'Gemini'])
    
    plt.title('Model Agreement Correlation Matrix', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'model_agreement.png'))
    print(f"\nModel agreement plot saved to {os.path.join(FIGURES_DIR, 'model_agreement.png')}")

def analyze_extreme_cases(df):
    """Analyze cases with extreme disagreement"""
    # Calculate absolute error
    df['abs_error'] = abs(df['error'])
    
    # Find cases with large disagreement
    large_errors = df[df['abs_error'] >= 2].sort_values('abs_error', ascending=False)
    
    print("\nCases with large disagreement (|error| >= 2):")
    print("\nTop 10 largest disagreements:")
    for _, row in large_errors.head(10).iterrows():
        print(f"\nSource: {row['source_outlet']}")
        print(f"Title: {row['title']}")
        print(f"Model: {row['model']}")
        print(f"AllSides Rating: {row['source_rating']}")
        print(f"AI Rating: {row['ai_rating']}")
        print(f"Error: {row['error']:.2f}")

def main():
    """Main function to analyze patterns"""
    # Load and merge data
    df = load_and_merge_data()
    
    # Analyze patterns
    analyze_bias_by_outlet(df)
    analyze_error_patterns(df)
    analyze_agreement_patterns(df)
    analyze_extreme_cases(df)

if __name__ == "__main__":
    main() 