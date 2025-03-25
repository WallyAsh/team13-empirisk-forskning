#!/usr/bin/env python3
"""
Visualize political ratings from DeepSeek classifier with precise AllSides source ratings.
Creates scatter plots to show the correlation between AI ratings and precise AllSides ratings.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from adjustText import adjust_text

# File paths
DEEPSEEK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "deepseek")
DATA_FILE = os.path.join(DEEPSEEK_DIR, "deepseek_articles_updated.json")
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

def create_precise_comparison_plot(data):
    """Create a scatter plot comparing AI ratings to precise source ratings"""
    
    # Extract data for visualization
    ai_ratings = []
    source_precise_ratings = []
    source_outlets = []
    source_categories = []
    
    for article in data:
        if ('ai_political_rating' in article and article['ai_political_rating'] is not None and
            'source_rating_value' in article and article['source_rating_value'] is not None):
            ai_ratings.append(article['ai_political_rating'])
            source_precise_ratings.append(article['source_rating_value'])
            source_outlets.append(article.get('source_outlet', 'Unknown'))
            source_categories.append(article.get('source_rating', 'Unknown'))
    
    if not ai_ratings:
        print("No valid data found for precise comparison")
        return
    
    print(f"Creating precise comparison plot for {len(ai_ratings)} articles")
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame({
        'ai_rating': ai_ratings,
        'source_rating': source_precise_ratings,
        'source_outlet': source_outlets,
        'source_category': source_categories
    })
    
    # Calculate differences for each source
    df['difference'] = df['ai_rating'] - df['source_rating']
    df['abs_difference'] = abs(df['difference'])
    
    # Create scatter plot
    plt.figure(figsize=(14, 12))
    
    # Color map based on source category
    colors = [CATEGORY_COLORS.get(cat, "#000000") for cat in df['source_category']]
    
    # Create scatter plot
    scatter = plt.scatter(df['source_rating'], df['ai_rating'], c=colors, alpha=0.7, s=50)
    
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
        'source_rating': 'mean',
        'source_category': 'first'
    }).reset_index()
    
    # Flatten the column structure
    outlet_stats.columns = ['source_outlet', 'mean_diff', 'mean_abs_diff', 'mean_ai', 'count', 'mean_source', 'source_category']
    
    # Filter to outlets with at least 3 articles
    major_outlets = outlet_stats[outlet_stats['count'] >= 3].sort_values('count', ascending=False)
    
    # Label major outlets on the plot
    texts = []
    for i, row in major_outlets.iterrows():
        # Only label outlets with significant count or disagreement
        if row['count'] >= 5 or abs(row['mean_diff']) > 1.5:
            txt = plt.text(row['mean_source'], row['mean_ai'], 
                     row['source_outlet'], 
                     fontsize=9, fontweight='bold')
            texts.append(txt)
    
    # Adjust text to avoid overlap
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))
    
    # Calculate correlation and errors
    correlation = np.corrcoef(df['source_rating'], df['ai_rating'])[0, 1]
    r_squared = correlation**2
    mean_error = df['difference'].mean()
    mean_abs_error = df['abs_difference'].mean()
    
    # Calculate root mean squared error (RMSE)
    rmse = np.sqrt(np.mean(df['difference']**2))
    
    # Calculate rating difference categories
    small_diff = sum(abs(diff) < 1 for diff in df['difference'])
    medium_diff = sum((abs(diff) >= 1) & (abs(diff) < 2) for diff in df['difference'])
    large_diff = sum(abs(diff) >= 2 for diff in df['difference'])
    
    # Calculate directional accuracy (same sign)
    same_direction = sum((source * ai > 0) for source, ai in zip(df['source_rating'], df['ai_rating']) 
                         if abs(source) > 0.5 and abs(ai) > 0.5)
    not_center_count = sum((abs(source) > 0.5) & (abs(ai) > 0.5) for source, ai in zip(df['source_rating'], df['ai_rating']))
    directional_accuracy = (same_direction / not_center_count * 100) if not_center_count > 0 else 0
    
    # Calculate categorical accuracy (within same category range)
    same_category = 0
    for idx, row in df.iterrows():
        source_cat = get_category_from_rating(row['source_rating'])
        ai_cat = get_category_from_rating(row['ai_rating'])
        if source_cat == ai_cat:
            same_category += 1
    
    category_accuracy = (same_category / len(df) * 100)
    
    # Add statistics to plot
    stats_text = (
        f"Correlation: {correlation:.2f} (R² = {r_squared:.2f})\n"
        f"Mean Error: {mean_error:.2f}\n"
        f"Mean Abs Error: {mean_abs_error:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"Same Category: {same_category}/{len(df)} ({category_accuracy:.1f}%)\n"
        f"Same Direction: {same_direction}/{not_center_count} ({directional_accuracy:.1f}%)\n"
        f"Small Diff (<1): {small_diff} ({small_diff/len(df)*100:.1f}%)\n"
        f"Medium Diff (1-2): {medium_diff} ({medium_diff/len(df)*100:.1f}%)\n"
        f"Large Diff (>2): {large_diff} ({large_diff/len(df)*100:.1f}%)"
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
    plt.title(f"AI Rating vs Precise AllSides Rating", fontsize=16)
    plt.xlabel("AllSides Precise Rating (-6 to 6)", fontsize=14)
    plt.ylabel("AI Rating (-6 to 6)", fontsize=14)
    
    # Set axis limits
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    
    # Add grid
    plt.grid(alpha=0.2)
    
    # Save the plot
    output_file = os.path.join(FIGURES_DIR, "ai_vs_precise_source_ratings.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Precise comparison plot saved to {output_file}")
    
    # Show the plot
    plt.show()
    
    # Create a regression plot to better visualize the relationship
    plt.figure(figsize=(12, 10))
    sns.regplot(x='source_rating', y='ai_rating', data=df, scatter_kws={'alpha':0.5, 'color':'blue'}, line_kws={'color':'red'})
    
    plt.title(f"AI Rating vs Precise AllSides Rating (Correlation: {correlation:.2f})", fontsize=16)
    plt.xlabel("AllSides Precise Rating (-6 to 6)", fontsize=14)
    plt.ylabel("AI Rating (-6 to 6)", fontsize=14)
    plt.grid(alpha=0.2)
    
    # Add vertical and horizontal lines at category boundaries
    for boundary in [-3, -1, 1, 3]:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
        plt.axhline(y=boundary, color='gray', linestyle='--', alpha=0.3)
    
    # Add diagonal line (perfect agreement)
    plt.plot([-6, 6], [-6, 6], 'k--', alpha=0.5)
    
    # Save the regression plot
    reg_output_file = os.path.join(FIGURES_DIR, "ai_vs_precise_regression.png")
    plt.savefig(reg_output_file, dpi=300, bbox_inches='tight')
    print(f"Regression plot saved to {reg_output_file}")
    
    # Show the plot
    plt.show()
    
    # Return statistics for reporting
    return {
        'correlation': correlation,
        'r_squared': r_squared,
        'mean_error': mean_error,
        'mean_abs_error': mean_abs_error,
        'rmse': rmse,
        'category_accuracy': category_accuracy,
        'directional_accuracy': directional_accuracy
    }

def create_category_boxplot(data):
    """Create boxplots of AI ratings by source category"""
    # Extract data
    records = []
    
    for article in data:
        if 'ai_political_rating' in article and article['ai_political_rating'] is not None:
            source_category = article.get('source_rating', 'Unknown')
            if source_category != 'Unknown' and source_category != 'Not rated':
                records.append({
                    'source_category': source_category,
                    'ai_rating': article['ai_political_rating'],
                    'source_precise': article.get('source_rating_value')
                })
    
    if not records:
        print("No valid data for boxplot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Order categories from left to right
    category_order = ['Left', 'Lean Left', 'Center', 'Lean Right', 'Right']
    df['source_category'] = pd.Categorical(df['source_category'], categories=category_order, ordered=True)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create boxplot
    sns.boxplot(x='source_category', y='ai_rating', data=df, palette=CATEGORY_COLORS)
    
    # Add swarmplot for individual points
    sns.swarmplot(x='source_category', y='ai_rating', data=df, size=7, color='black', alpha=0.5)
    
    # Add horizontal lines at category boundaries
    for boundary in [-3, -1, 1, 3]:
        plt.axhline(y=boundary, color='gray', linestyle='--', alpha=0.5)
    
    # Add line for each category's precise average rating
    category_precise_means = df.groupby('source_category')['source_precise'].mean()
    
    for i, category in enumerate(category_order):
        if category in category_precise_means:
            precise_mean = category_precise_means[category]
            plt.plot([i-0.4, i+0.4], [precise_mean, precise_mean], 'r-', linewidth=2)
    
    # Add statistics
    category_stats = df.groupby('source_category').agg({
        'ai_rating': ['mean', 'median', 'std', 'count'],
        'source_precise': ['mean']
    })
    
    # Flatten multi-level column names
    category_stats.columns = ['ai_mean', 'ai_median', 'ai_std', 'count', 'source_mean']
    
    # Add annotation box with stats
    stats_text = "Category Statistics:\n" + "\n".join([
        f"{cat}: n={stats['count']}, AI mean={stats['ai_mean']:.2f}, Source mean={stats['source_mean']:.2f}, "
        f"Diff={stats['ai_mean']-stats['source_mean']:.2f}"
        for cat, stats in category_stats.iterrows()
    ])
    
    plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='figure fraction',
                 ha='left', va='bottom', bbox=dict(boxstyle='round', fc='white', alpha=0.7),
                 fontsize=10)
    
    # Set labels and title
    plt.title('AI Political Ratings by Source Category', fontsize=16)
    plt.xlabel('Source Category (AllSides)', fontsize=14)
    plt.ylabel('AI Political Rating (-6 to 6)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    # Add category ranges
    for cat, (min_val, max_val) in CATEGORY_RANGES.items():
        plt.fill_between([-0.5, 4.5], [min_val, min_val], [max_val, max_val], alpha=0.05, color=CATEGORY_COLORS.get(cat, '#CCCCCC'))
        plt.text(4.6, (min_val + max_val) / 2, cat, va='center', fontsize=10)
    
    # Save figure
    output_file = os.path.join(FIGURES_DIR, "ai_ratings_by_category.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Category boxplot saved to {output_file}")
    
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
        # Check if we have precise ratings
        precise_ratings_count = sum(1 for article in data if 'source_rating_value' in article and article['source_rating_value'] is not None)
        
        if precise_ratings_count == 0:
            print("No source rating values found in the data.")
            print(f"Expected data file: {DATA_FILE}")
            return
        
        print(f"Found {precise_ratings_count} articles with source rating values")
        
        # Generate visualizations
        stats = create_precise_comparison_plot(data)
        create_category_boxplot(data)
        
        # Display summary of findings
        if stats:
            print("\n====== SUMMARY OF FINDINGS ======")
            print(f"Correlation between AI and AllSides: {stats['correlation']:.2f} (R² = {stats['r_squared']:.2f})")
            print(f"Mean error: {stats['mean_error']:.2f}")
            print(f"Mean absolute error: {stats['mean_abs_error']:.2f}")
            print(f"Root mean squared error: {stats['rmse']:.2f}")
            print(f"Same category accuracy: {stats['category_accuracy']:.1f}%")
            print(f"Same direction accuracy: {stats['directional_accuracy']:.1f}%")
            print("================================")
        
        print("All visualizations completed!")

if __name__ == "__main__":
    main() 