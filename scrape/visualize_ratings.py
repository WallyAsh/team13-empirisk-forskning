#!/usr/bin/env python3
"""
Visualize political ratings from DeepSeek classifier results
Creates scatter plots to show the distribution of political leanings
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from matplotlib.lines import Line2D
from collections import defaultdict
import seaborn as sns
from adjustText import adjust_text

# File paths
DEEPSEEK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "deepseek")
DATA_FILE = os.path.join(DEEPSEEK_DIR, "deepseek_articles.json")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Create figures directory if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)

# Define category ranges for visualization
CATEGORY_RANGES = {
    "Left": (-6, -3),
    "Lean Left": (-3, -1),
    "Center": (-1, 1),
    "Lean Right": (1, 3),
    "Right": (3, 6)
}

# Define color mapping for categories
CATEGORY_COLORS = {
    "Left": "#3333FF",       # Blue
    "Lean Left": "#99CCFF",  # Light blue
    "Center": "#AAAAAA",     # Grey
    "Lean Right": "#FFCC99", # Light red
    "Right": "#FF3333"       # Red
}

def load_data():
    """Load articles data from JSON file"""
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} articles from {DATA_FILE}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def create_distribution_histogram(data):
    """Create a histogram of political ratings distribution"""
    # Extract data for visualization
    ratings = []
    categories = []
    
    for article in data:
        if 'ai_political_rating' in article and article['ai_political_rating'] is not None:
            ratings.append(article['ai_political_rating'])
            categories.append(article.get('ai_political_leaning', 'Unknown'))
    
    if not ratings:
        print("No rating data found to visualize histogram")
        return
    
    print(f"Creating histogram for {len(ratings)} articles with ratings")
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Create a color map for categories
    category_count = defaultdict(int)
    for cat in categories:
        if cat in CATEGORY_COLORS:
            category_count[cat] += 1
    
    # Plot histogram with category colors
    sns.histplot(ratings, bins=24, kde=True)
    
    # Add background shading for categories
    for category, (min_val, max_val) in CATEGORY_RANGES.items():
        color = CATEGORY_COLORS[category]
        plt.axvspan(min_val, max_val, alpha=0.1, color=color)
        # Add category label with count
        count = category_count.get(category, 0)
        percentage = (count / len(ratings) * 100) if ratings else 0
        plt.text((min_val + max_val) / 2, -0.5, f"{category}\n{count} ({percentage:.1f}%)", 
                ha='center', fontsize=10)
    
    # Add vertical lines at category boundaries
    for boundary in [-3, -1, 1, 3]:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Set plot title and labels
    plt.title("Distribution Histogram of Political Ratings by DeepSeek AI")
    plt.xlabel("Political Rating Scale (-6 to 6)")
    plt.ylabel("Number of Articles")
    
    # Calculate statistics
    mean_rating = np.mean(ratings)
    median_rating = np.median(ratings)
    std_rating = np.std(ratings)
    
    # Add statistical annotations
    stats_text = (f"Mean: {mean_rating:.2f}\nMedian: {median_rating:.2f}\n"
                 f"Std Dev: {std_rating:.2f}\nTotal: {len(ratings)}")
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    
    # Set X-axis limits to match our scale
    plt.xlim(-6.5, 6.5)
    
    # Add mean and median lines
    plt.axvline(x=mean_rating, color='black', linestyle='-', alpha=0.7, label=f'Mean ({mean_rating:.2f})')
    plt.axvline(x=median_rating, color='black', linestyle=':', alpha=0.7, label=f'Median ({median_rating:.2f})')
    plt.legend()
    
    # Save the plot
    output_file = os.path.join(FIGURES_DIR, "rating_distribution_histogram.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {output_file}")
    
    # Show the plot
    plt.show()

def create_scatter_plot(data):
    """Create a scatter plot of political ratings by source"""
    # Extract data for visualization
    ratings = []
    source_ratings = []
    source_outlets = []
    
    for article in data:
        if 'ai_political_rating' in article and article['ai_political_rating'] is not None:
            ratings.append(article['ai_political_rating'])
            source_ratings.append(article.get('source_rating', 'Unknown'))
            source_outlets.append(article.get('source_outlet', 'Unknown'))
    
    if not ratings:
        print("No rating data found to visualize")
        return
    
    print(f"Creating scatter plot for {len(ratings)} articles with ratings")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame({
        'rating': ratings,
        'source_rating': source_ratings,
        'source_outlet': source_outlets
    })
    
    # Create scatter plot
    plt.figure(figsize=(14, 10))
    
    # Create a mapping of source ratings to colors
    color_map = {source: CATEGORY_COLORS.get(source, "#000000") for source in df['source_rating'].unique()}
    
    # Group by source outlet to calculate mean ratings
    outlet_stats = df.groupby('source_outlet').agg({
        'rating': ['mean', 'count', 'std']
    }).reset_index()
    outlet_stats.columns = ['source_outlet', 'mean_rating', 'count', 'std_rating']
    
    # Sort by count (descending) to highlight major sources
    outlet_stats = outlet_stats.sort_values('count', ascending=False)
    top_outlets = outlet_stats[outlet_stats['count'] >= 5]  # Only outlets with 5+ articles
    
    # Create scatter plot with top sources labeled
    for source in top_outlets['source_outlet']:
        source_df = df[df['source_outlet'] == source]
        mean_rating = source_df['rating'].mean()
        source_category = source_df['source_rating'].iloc[0]
        color = color_map.get(source_category, "#000000")
        count = len(source_df)
        # Plot the mean position with size proportional to count
        plt.scatter(mean_rating, 0, color=color, alpha=0.9, s=count*20, 
                    edgecolors='black', linewidths=1, zorder=5)
        # Add source label
        plt.text(mean_rating, 0.02, source, ha='center', va='bottom', 
                fontsize=9, rotation=45, fontweight='bold')
    
    # Add background shading for categories
    for category, (min_val, max_val) in CATEGORY_RANGES.items():
        color = CATEGORY_COLORS[category]
        plt.axvspan(min_val, max_val, alpha=0.1, color=color)
        # Add category label
        plt.text((min_val + max_val) / 2, -0.15, category, ha='center', fontsize=10)
    
    # Add grid lines at category boundaries
    for boundary in [-3, -1, 1, 3]:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Create legend for source ratings
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor=color, markersize=10, label=category)
                      for category, color in CATEGORY_COLORS.items()]
    
    plt.legend(handles=legend_elements, title="Source AllSides Rating", loc='upper left')
    
    # Set plot title and labels
    plt.title("Political Ratings by News Source (DeepSeek AI)")
    plt.xlabel("Political Rating Scale (-6 to 6)")
    plt.ylabel("")
    plt.yticks([])  # Hide Y-axis ticks
    
    # Set X-axis limits to match our scale
    plt.xlim(-6.5, 6.5)
    plt.ylim(-0.2, 0.5)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Save the plot
    output_file = os.path.join(FIGURES_DIR, "source_rating_distribution.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Source scatter plot saved to {output_file}")
    
    # Show the plot
    plt.show()

def create_source_comparison_plot(data):
    """Create a scatter plot comparing AI ratings to source ratings"""
    # Map source ratings to numerical values for comparison
    SOURCE_RATING_VALUES = {
        "Left": -4.5,
        "Lean Left": -2.0, 
        "Center": 0,
        "Lean Right": 2.0,
        "Right": 4.5,
        "Unknown": None,
        "Not rated": None
    }
    
    # Extract data for visualization
    ai_ratings = []
    source_numerical_ratings = []
    source_outlets = []
    source_categories = []
    
    for article in data:
        if 'ai_political_rating' in article and article['ai_political_rating'] is not None:
            source_rating = article.get('source_rating', 'Unknown')
            if source_rating != 'Unknown' and source_rating != 'Not rated':
                ai_ratings.append(article['ai_political_rating'])
                source_numerical_ratings.append(SOURCE_RATING_VALUES[source_rating])
                source_outlets.append(article.get('source_outlet', 'Unknown'))
                source_categories.append(source_rating)
    
    if not ai_ratings:
        print("No valid data found for source comparison")
        return
    
    print(f"Creating source comparison plot for {len(ai_ratings)} articles")
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame({
        'ai_rating': ai_ratings,
        'source_rating': source_numerical_ratings,
        'source_outlet': source_outlets,
        'source_category': source_categories
    })
    
    # Calculate differences for each source
    df['difference'] = df['ai_rating'] - df['source_rating']
    df['abs_difference'] = abs(df['difference'])
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Color map based on source category
    colors = [CATEGORY_COLORS.get(cat, "#000000") for cat in df['source_category']]
    
    # Create scatter plot
    plt.scatter(df['source_rating'], df['ai_rating'], c=colors, alpha=0.5, s=40)
    
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
    outlet_stats = df.groupby('source_outlet').agg({
        'difference': 'mean',
        'abs_difference': 'mean',
        'ai_rating': ['mean', 'count'],
        'source_rating': 'mean'
    }).reset_index()
    
    # Flatten the column structure
    outlet_stats.columns = ['source_outlet', 'mean_diff', 'mean_abs_diff', 'mean_ai', 'count', 'mean_source']
    
    # Filter to outlets with at least 3 articles
    major_outlets = outlet_stats[outlet_stats['count'] >= 3].sort_values('mean_abs_diff', ascending=False)
    
    # Label major outlets on the plot
    texts = []
    for i, row in major_outlets.iterrows():
        if abs(row['mean_diff']) > 1.5:  # Only label outlets with significant disagreement
            txt = plt.text(row['mean_source'], row['mean_ai'], 
                     row['source_outlet'],
                     fontsize=8, fontweight='bold')
            texts.append(txt)
    
    # Adjust text to avoid overlap
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
    
    # Calculate correlation and errors
    correlation = np.corrcoef(df['source_rating'], df['ai_rating'])[0, 1]
    mean_error = df['difference'].mean()
    mean_abs_error = df['abs_difference'].mean()
    
    # Count articles by difference magnitude
    small_diff = sum(abs(diff) < 2 for diff in df['difference'])
    medium_diff = sum((abs(diff) >= 2) & (abs(diff) < 4) for diff in df['difference'])
    large_diff = sum(abs(diff) >= 4 for diff in df['difference'])
    
    # Calculate percentage agreement
    same_direction = sum((source * ai >= 0) for source, ai in zip(df['source_rating'], df['ai_rating']) 
                         if abs(source) > 0.5 and abs(ai) > 0.5)
    not_center_count = sum((abs(source) > 0.5) & (abs(ai) > 0.5) for source, ai in zip(df['source_rating'], df['ai_rating']))
    directional_accuracy = (same_direction / not_center_count * 100) if not_center_count > 0 else 0
    
    # Same category count
    same_category = 0
    for idx, row in df.iterrows():
        ai_cat = get_category_from_rating(row['ai_rating'])
        source_cat = row['source_category']
        if ai_cat == source_cat:
            same_category += 1
    
    category_accuracy = (same_category / len(df) * 100)
    
    # Add statistics to plot
    stats_text = (
        f"Correlation: {correlation:.2f}\n"
        f"Mean Error: {mean_error:.2f}\n"
        f"Mean Abs Error: {mean_abs_error:.2f}\n"
        f"Same Category: {same_category}/{len(df)} ({category_accuracy:.1f}%)\n"
        f"Same Direction: {same_direction}/{not_center_count} ({directional_accuracy:.1f}%)\n"
        f"Small Diff (<2): {small_diff} ({small_diff/len(df)*100:.1f}%)\n"
        f"Medium Diff (2-4): {medium_diff} ({medium_diff/len(df)*100:.1f}%)\n"
        f"Large Diff (>4): {large_diff} ({large_diff/len(df)*100:.1f}%)"
    )
    
    plt.annotate(stats_text, xy=(0.03, 0.97), xycoords='axes fraction',
                 ha='left', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    
    # Create legend for source categories
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=color, markersize=10, label=category)
                     for category, color in CATEGORY_COLORS.items()]
    
    # Add diagonal band for "close enough" agreement
    plt.fill_between([-6, 6], [-8, 4], [-4, 8], color='green', alpha=0.05)
    
    # Add legend
    plt.legend(handles=legend_elements, title="Source Category", loc='lower right')
    
    # Set plot title and labels
    plt.title(f"AI Rating vs Source Rating")
    plt.xlabel("Source Rating (AllSides)")
    plt.ylabel("AI Rating (DeepSeek)")
    
    # Set axis limits
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    
    # Add grid
    plt.grid(alpha=0.2)
    
    # Save the plot
    output_file = os.path.join(FIGURES_DIR, "ai_vs_source_ratings.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")
    
    # Show the plot
    plt.show()

def create_source_accuracy_bar_chart(data):
    """Create a bar chart showing AI classification accuracy by source category"""
    # Map source ratings to numerical values
    SOURCE_RATING_VALUES = {
        "Left": -4.5,
        "Lean Left": -2.0, 
        "Center": 0,
        "Lean Right": 2.0,
        "Right": 4.5,
        "Unknown": None,
        "Not rated": None
    }
    
    # Extract data
    results = []
    for article in data:
        if 'ai_political_rating' in article and 'source_rating' in article:
            if article['source_rating'] != 'Unknown' and article['source_rating'] != 'Not rated':
                source_cat = article['source_rating']
                source_val = SOURCE_RATING_VALUES[source_cat]
                ai_val = article['ai_political_rating']
                ai_cat = get_category_from_rating(ai_val)
                
                results.append({
                    'source_category': source_cat,
                    'ai_category': ai_cat,
                    'source_rating': source_val,
                    'ai_rating': ai_val,
                    'outlet': article.get('source_outlet', 'Unknown'),
                    'exact_match': source_cat == ai_cat,
                    'close_match': abs(source_val - ai_val) < 2,
                    'same_direction': (source_val * ai_val > 0) if source_val != 0 and ai_val != 0 else False
                })
    
    if not results:
        print("No data available for accuracy analysis")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate accuracy metrics by source category
    accuracy_by_category = df.groupby('source_category').agg({
        'exact_match': 'mean',
        'close_match': 'mean',
        'same_direction': 'mean',
        'source_category': 'count'
    }).rename(columns={'source_category': 'count'})
    
    # Sort by source category from Left to Right
    category_order = ["Left", "Lean Left", "Center", "Lean Right", "Right"]
    accuracy_by_category = accuracy_by_category.reindex(category_order)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Set bar width
    bar_width = 0.25
    
    # Set bar positions
    r1 = np.arange(len(accuracy_by_category))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    plt.bar(r1, accuracy_by_category['exact_match']*100, width=bar_width, 
           color='#4CAF50', label='Exact Match')
    plt.bar(r2, accuracy_by_category['close_match']*100, width=bar_width, 
           color='#2196F3', label='Close Match (<2 diff)')
    plt.bar(r3, accuracy_by_category['same_direction']*100, width=bar_width, 
           color='#FF9800', label='Same Direction')
    
    # Add labels
    plt.xlabel('Source Category')
    plt.ylabel('Accuracy (%)')
    plt.title('AI Classification Accuracy by Source Category')
    plt.xticks([r + bar_width for r in range(len(accuracy_by_category))], 
              accuracy_by_category.index)
    
    # Add count labels above bars
    for i, count in enumerate(accuracy_by_category['count']):
        plt.text(i, 5, f"n={count}", ha='center', va='bottom', fontsize=9)
    
    # Add percentage labels on bars
    for i, val in enumerate(accuracy_by_category['exact_match']):
        plt.text(r1[i], val*100 + 2, f"{val*100:.1f}%", ha='center', va='bottom', fontsize=8)
    
    for i, val in enumerate(accuracy_by_category['close_match']):
        plt.text(r2[i], val*100 + 2, f"{val*100:.1f}%", ha='center', va='bottom', fontsize=8)
        
    for i, val in enumerate(accuracy_by_category['same_direction']):
        plt.text(r3[i], val*100 + 2, f"{val*100:.1f}%", ha='center', va='bottom', fontsize=8)
    
    # Add overall accuracy
    overall_exact = df['exact_match'].mean() * 100
    overall_close = df['close_match'].mean() * 100
    overall_direction = df['same_direction'].mean() * 100
    
    plt.axhline(y=overall_exact, color='#4CAF50', linestyle='--', alpha=0.7)
    plt.axhline(y=overall_close, color='#2196F3', linestyle='--', alpha=0.7)
    plt.axhline(y=overall_direction, color='#FF9800', linestyle='--', alpha=0.7)
    
    plt.text(4.5, overall_exact, f"Overall: {overall_exact:.1f}%", color='#4CAF50', fontweight='bold')
    plt.text(4.5, overall_close, f"Overall: {overall_close:.1f}%", color='#2196F3', fontweight='bold')
    plt.text(4.5, overall_direction, f"Overall: {overall_direction:.1f}%", color='#FF9800', fontweight='bold')
    
    # Add grid and legend
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # Set y-axis limit
    plt.ylim(0, 100)
    
    # Save plot
    output_file = os.path.join(FIGURES_DIR, "accuracy_by_category.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Accuracy chart saved to {output_file}")
    
    # Show plot
    plt.show()

def get_category_from_rating(rating):
    """Convert numerical rating to category string"""
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

def main():
    """Main function to generate all visualizations"""
    data = load_data()
    if data:
        create_distribution_histogram(data)
        create_scatter_plot(data)
        create_source_comparison_plot(data)
        create_source_accuracy_bar_chart(data)
        print("All visualizations completed!")

if __name__ == "__main__":
    main() 