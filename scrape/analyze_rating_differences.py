#!/usr/bin/env python3
"""
Analyze rating differences between AI and source ratings in DeepSeek articles.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the bias mapping for reference
BIAS_MAPPING = {
    'Left': -2,
    'Lean Left': -1,
    'Center': 0,
    'Lean Right': 1,
    'Right': 2
}

def load_deepseek_articles():
    """Load the DeepSeek articles data."""
    file_path = 'models/deepseek/deepseek_articles.json'
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_rating_differences():
    """Analyze and visualize rating differences between AI and source ratings."""
    # Load the data
    articles = load_deepseek_articles()
    df = pd.DataFrame(articles)
    
    # Filter out articles with unavailable full text
    df = df[df['full_text'] != "Not available..."]
    print(f"\nTotal articles after filtering unavailable text: {len(df)}")
    
    # Filter out articles with no rating difference
    df_with_diffs = df[df['rating_difference'].notna()]
    
    # Count rating differences
    diff_counts = df_with_diffs['rating_difference'].value_counts().sort_index()
    
    # Calculate percentages
    total_analyzable = len(df_with_diffs)
    diff_percentages = (diff_counts / total_analyzable * 100).round(1)
    
    # Print detailed analysis
    print("\nDetailed Rating Difference Analysis")
    print("===================================")
    print(f"Total analyzable articles: {total_analyzable}")
    print("\nRating Difference Distribution:")
    print("--------------------------------")
    
    interpretations = {
        -4: "AI classified as Left when source was Right",
        -3: "AI classified as Left when source was Lean Right",
        -2: "AI classified as Left when source was Center, or Lean Left when source was Right",
        -1: "AI classified as Left when source was Lean Left, or Lean Left when source was Center, or Center when source was Lean Right",
        0: "Perfect match",
        1: "AI classified as Right when source was Lean Right, or Lean Right when source was Center, or Center when source was Lean Left",
        2: "AI classified as Right when source was Center, or Lean Right when source was Left",
        3: "AI classified as Right when source was Lean Left",
        4: "AI classified as Right when source was Left"
    }
    
    for diff in sorted(diff_counts.index):
        count = diff_counts[diff]
        percentage = diff_percentages[diff]
        interpretation = interpretations.get(diff, "Unknown pattern")
        print(f"\nDifference of {diff}:")
        print(f"  Count: {count} articles")
        print(f"  Percentage: {percentage}%")
        print(f"  Interpretation: {interpretation}")
    
    # Calculate additional statistics
    print("\nOverall Statistics")
    print("-----------------")
    print(f"Mean difference: {df_with_diffs['rating_difference'].mean():.2f}")
    print(f"Median difference: {df_with_diffs['rating_difference'].median():.2f}")
    print(f"Standard deviation: {df_with_diffs['rating_difference'].std():.2f}")
    print(f"Perfect matches (0): {diff_counts.get(0, 0)} articles ({diff_percentages.get(0, 0):.1f}%)")
    
    # Calculate directional accuracy (left vs right)
    df_with_diffs['source_direction'] = df_with_diffs['source_rating_value'].apply(lambda x: 'Left' if x < 0 else 'Right' if x > 0 else 'Center')
    df_with_diffs['ai_direction'] = df_with_diffs['ai_rating_value'].apply(lambda x: 'Left' if x < 0 else 'Right' if x > 0 else 'Center')
    df_with_diffs['direction_match'] = df_with_diffs['source_direction'] == df_with_diffs['ai_direction']
    
    directional_accuracy = df_with_diffs['direction_match'].mean() * 100
    print(f"\nDirectional Accuracy (Left vs Right): {directional_accuracy:.1f}%")
    
    # Create visualizations
    create_visualizations(df_with_diffs, diff_counts, diff_percentages)

def create_visualizations(df, diff_counts, diff_percentages):
    """Create visualizations of the rating differences."""
    # Create a directory for the visualizations
    os.makedirs('figures', exist_ok=True)
    
    # 1. Bar plot of rating differences
    plt.figure(figsize=(12, 6))
    bars = plt.bar(diff_counts.index, diff_counts.values, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        x = bar.get_x()
        count = int(height)
        percentage = diff_percentages.get(x, 0)
        plt.text(x + bar.get_width()/2., height,
                f'{count}\n({percentage:.1f}%)',
                ha='center', va='bottom')
    
    plt.title('Distribution of Rating Differences\n(AI Rating - Source Rating)')
    plt.xlabel('Rating Difference')
    plt.ylabel('Number of Articles')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figures/rating_differences_distribution.png', dpi=300)
    plt.close()
    
    # 2. Heatmap of AI vs Source ratings
    plt.figure(figsize=(10, 8))
    pivot_table = pd.crosstab(df['source_rating_value'], df['ai_rating_value'])
    sns.heatmap(pivot_table, annot=True, fmt='d', cmap='Blues')
    plt.title('AI vs Source Rating Distribution')
    plt.xlabel('AI Rating Value')
    plt.ylabel('Source Rating Value')
    plt.tight_layout()
    plt.savefig('figures/rating_heatmap.png', dpi=300)
    plt.close()

def main():
    """Main function to run the analysis."""
    print("Starting rating difference analysis...")
    analyze_rating_differences()
    print("\nAnalysis complete! Visualizations saved to the 'figures' directory.")

if __name__ == "__main__":
    main() 