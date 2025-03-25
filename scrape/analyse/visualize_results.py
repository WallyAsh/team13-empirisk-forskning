#!/usr/bin/env python3
"""
Script to visualize the classification results from the selected subset.

This script:
1. Loads the classified subset results
2. Generates various visualizations to analyze the results
3. Saves the visualizations to the figures directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MultipleLocator

# File paths
DEFAULT_RESULTS_PATH = 'analysis_subset/classification_results.csv'
DEFAULT_FIGURES_DIR = 'analysis_subset/figures'

def load_results(file_path):
    """Load classification results from CSV file"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def ensure_figures_dir(dir_path):
    """Ensure figures directory exists"""
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def plot_distribution_comparison(df, figures_dir):
    """Plot source vs AI distribution comparison"""
    # Count occurrences
    source_counts = df['source_rating'].value_counts().reindex(['Left', 'Lean Left', 'Center', 'Lean Right', 'Right'])
    ai_counts = df['ai_political_leaning'].value_counts().reindex(['Left', 'Lean Left', 'Center', 'Lean Right', 'Right'])
    
    # Calculate percentages
    total_articles = len(df)
    source_pct = (source_counts / total_articles) * 100
    ai_pct = (ai_counts / total_articles) * 100
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Source Rating': source_pct,
        'AI Classification': ai_pct
    })
    
    # Create plot
    plt.figure(figsize=(12, 8))
    comparison_df.plot(kind='bar', ax=plt.gca())
    plt.title('Source Rating vs AI Classification Distribution', fontsize=16)
    plt.xlabel('Political Leaning Category', fontsize=14)
    plt.ylabel('Percentage of Articles (%)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(figures_dir, 'distribution_comparison.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved distribution comparison to {fig_path}")

def plot_rating_distribution(df, figures_dir):
    """Plot numerical rating distribution"""
    # Filter out None values
    ratings_df = df[df['ai_political_rating'].notna()].copy()
    
    if len(ratings_df) == 0:
        print("No valid ratings to plot")
        return
    
    # Convert to numeric
    ratings_df['ai_political_rating'] = pd.to_numeric(ratings_df['ai_political_rating'])
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    sns.histplot(ratings_df['ai_political_rating'], bins=24, kde=True)
    
    # Add vertical lines for category boundaries
    plt.axvline(x=-3, color='r', linestyle='--', alpha=0.7, label='Left/Lean Left boundary')
    plt.axvline(x=-1, color='g', linestyle='--', alpha=0.7, label='Lean Left/Center boundary')
    plt.axvline(x=1, color='g', linestyle='--', alpha=0.7, label='Center/Lean Right boundary')
    plt.axvline(x=3, color='r', linestyle='--', alpha=0.7, label='Lean Right/Right boundary')
    
    # Add category labels
    plt.text(-4.5, plt.gca().get_ylim()[1] * 0.9, "Left", fontsize=12)
    plt.text(-2, plt.gca().get_ylim()[1] * 0.9, "Lean Left", fontsize=12)
    plt.text(0, plt.gca().get_ylim()[1] * 0.9, "Center", fontsize=12)
    plt.text(2, plt.gca().get_ylim()[1] * 0.9, "Lean Right", fontsize=12)
    plt.text(4, plt.gca().get_ylim()[1] * 0.9, "Right", fontsize=12)
    
    plt.title('Distribution of AI Political Ratings', fontsize=16)
    plt.xlabel('Political Rating (-6 to 6)', fontsize=14)
    plt.ylabel('Number of Articles', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(-6.5, 6.5)
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(figures_dir, 'rating_distribution.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved rating distribution to {fig_path}")

def plot_confusion_matrix(df, figures_dir):
    """Plot confusion matrix of source vs AI classification"""
    # Create the confusion matrix
    categories = ['Left', 'Lean Left', 'Center', 'Lean Right', 'Right']
    conf_matrix = pd.DataFrame(0, index=categories, columns=categories)
    
    # Fill the confusion matrix
    for _, row in df.iterrows():
        source = row['source_rating']
        ai = row['ai_political_leaning']
        if source in categories and ai in categories:
            conf_matrix.loc[source, ai] += 1
    
    # Calculate percentages by row (source rating)
    conf_pct = conf_matrix.div(conf_matrix.sum(axis=1), axis=0) * 100
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix: Source Rating vs AI Classification', fontsize=16)
    plt.xlabel('AI Classification', fontsize=14)
    plt.ylabel('Source Rating', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(figures_dir, 'confusion_matrix.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {fig_path}")
    
    # Plot percentage version
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_pct, annot=True, fmt='.1f', cmap='Blues', cbar=True,
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix (% by Source): Source Rating vs AI Classification', fontsize=16)
    plt.xlabel('AI Classification', fontsize=14)
    plt.ylabel('Source Rating', fontsize=14)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(figures_dir, 'confusion_matrix_percent.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved percentage confusion matrix to {fig_path}")

def plot_outlet_accuracy(df, figures_dir):
    """Plot classification accuracy by news outlet"""
    # Get outlets with at least 3 articles
    outlet_counts = df['source_outlet'].value_counts()
    outlets_to_include = outlet_counts[outlet_counts >= 3].index.tolist()
    
    # Filter to selected outlets
    outlets_df = df[df['source_outlet'].isin(outlets_to_include)]
    
    # Calculate accuracy by outlet
    accuracy_by_outlet = outlets_df.groupby('source_outlet')['match_with_source'].agg(['count', 'mean'])
    accuracy_by_outlet['mean'] = accuracy_by_outlet['mean'] * 100  # Convert to percentage
    accuracy_by_outlet = accuracy_by_outlet.sort_values('mean', ascending=False)
    
    # Plot
    plt.figure(figsize=(14, 10))
    bars = plt.bar(accuracy_by_outlet.index, accuracy_by_outlet['mean'])
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', rotation=0)
    
    plt.title('Classification Accuracy by News Outlet', fontsize=16)
    plt.xlabel('News Outlet', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(figures_dir, 'outlet_accuracy.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved outlet accuracy to {fig_path}")

def plot_rating_difference(df, figures_dir):
    """Plot numerical rating difference (AI - Source)"""
    # Filter to rows with both ratings
    ratings_df = df[(df['ai_political_rating'].notna()) & (df['source_rating_value'].notna())].copy()
    
    if len(ratings_df) == 0:
        print("No valid rating pairs to plot")
        return
    
    # Convert to numeric
    ratings_df['ai_political_rating'] = pd.to_numeric(ratings_df['ai_political_rating'])
    ratings_df['source_rating_value'] = pd.to_numeric(ratings_df['source_rating_value'])
    
    # Calculate difference
    ratings_df['rating_difference'] = ratings_df['ai_political_rating'] - ratings_df['source_rating_value']
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    sns.histplot(ratings_df['rating_difference'], bins=20, kde=True)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.7, label='Perfect match')
    
    # Calculate statistics
    mean_diff = ratings_df['rating_difference'].mean()
    median_diff = ratings_df['rating_difference'].median()
    
    # Add statistics to the plot
    plt.axvline(x=mean_diff, color='g', linestyle='--', alpha=0.7, 
                label=f'Mean difference: {mean_diff:.2f}')
    
    plt.title('Distribution of Rating Differences (AI - Source)', fontsize=16)
    plt.xlabel('Rating Difference', fontsize=14)
    plt.ylabel('Number of Articles', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(figures_dir, 'rating_difference.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved rating difference to {fig_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize classification results")
    parser.add_argument('--results-path', default=DEFAULT_RESULTS_PATH,
                      help=f'Path to the results CSV file (default: {DEFAULT_RESULTS_PATH})')
    parser.add_argument('--figures-dir', default=DEFAULT_FIGURES_DIR,
                      help=f'Directory to save figures (default: {DEFAULT_FIGURES_DIR})')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_path}...")
    df = load_results(args.results_path)
    
    if df is None or len(df) == 0:
        print("No results to visualize. Exiting.")
        return
    
    print(f"Loaded {len(df)} articles")
    
    # Ensure figures directory exists
    figures_dir = ensure_figures_dir(args.figures_dir)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_distribution_comparison(df, figures_dir)
    plot_rating_distribution(df, figures_dir)
    plot_confusion_matrix(df, figures_dir)
    plot_outlet_accuracy(df, figures_dir)
    plot_rating_difference(df, figures_dir)
    
    print(f"\nAll visualizations saved to {figures_dir}")

if __name__ == "__main__":
    main() 