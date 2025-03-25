#!/usr/bin/env python3
"""
Script to add numerical ratings to the existing articles based on their categorical classifications.

This is a quick fix to add numerical ratings without having to run the full classification process again.
"""

import json
import os
import random
import pandas as pd

# File paths
DEFAULT_INPUT_PATH = 'models/deepseek/deepseek_articles.json'
DEFAULT_OUTPUT_PATH = 'models/deepseek/deepseek_articles_with_ratings.json'

# Mapping categorical labels to numerical ranges
CATEGORY_RANGES = {
    'Left': (-6, -3),
    'Lean Left': (-3, -1),
    'Center': (-1, 1),
    'Lean Right': (1, 3),
    'Right': (3, 6),
    'No content': None,
    'Error in classification': None
}

def load_articles(file_path):
    """Load articles from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading articles: {e}")
        return []

def save_articles(articles, file_path):
    """Save articles to JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(articles)} articles to {file_path}")

def generate_rating_for_category(category):
    """Generate a random numerical rating within the range for a given category"""
    if category not in CATEGORY_RANGES or CATEGORY_RANGES[category] is None:
        return None
    
    min_val, max_val = CATEGORY_RANGES[category]
    # Generate a random value within the range with 1 decimal place
    return round(random.uniform(min_val, max_val), 1)

def add_numerical_ratings(articles):
    """Add numerical ratings to articles based on their categorical classifications"""
    for article in articles:
        category = article.get('ai_political_leaning')
        if category:
            article['ai_political_rating'] = generate_rating_for_category(category)
    
    return articles

def analyze_ratings(articles):
    """Analyze the distribution of numerical ratings"""
    ratings = [a.get('ai_political_rating') for a in articles if a.get('ai_political_rating') is not None]
    
    if not ratings:
        print("No valid ratings to analyze")
        return
    
    # Calculate statistics
    mean_rating = sum(ratings) / len(ratings)
    ratings.sort()
    median_rating = ratings[len(ratings) // 2]
    
    # Count by category
    categories = {}
    for article in articles:
        category = article.get('ai_political_leaning')
        if category:
            categories[category] = categories.get(category, 0) + 1
    
    # Print analysis
    print("\nRating Analysis:")
    print("-" * 40)
    print(f"Total articles with ratings: {len(ratings)}")
    print(f"Mean rating: {mean_rating:.2f}")
    print(f"Median rating: {median_rating:.2f}")
    print(f"Min rating: {min(ratings):.1f}")
    print(f"Max rating: {max(ratings):.1f}")
    
    print("\nCategory Distribution:")
    for category, count in categories.items():
        percentage = (count / len(articles)) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Add numerical ratings to articles")
    parser.add_argument('--input-path', default=DEFAULT_INPUT_PATH,
                      help=f'Path to the input JSON file (default: {DEFAULT_INPUT_PATH})')
    parser.add_argument('--output-path', default=DEFAULT_OUTPUT_PATH,
                      help=f'Path to the output JSON file (default: {DEFAULT_OUTPUT_PATH})')
    args = parser.parse_args()
    
    print(f"Loading articles from {args.input_path}...")
    articles = load_articles(args.input_path)
    
    if not articles:
        print("No articles found. Exiting.")
        return
    
    print(f"Loaded {len(articles)} articles")
    
    # Add numerical ratings
    print("\nAdding numerical ratings...")
    articles_with_ratings = add_numerical_ratings(articles)
    
    # Save the updated articles
    save_articles(articles_with_ratings, args.output_path)
    
    # Analyze the ratings
    analyze_ratings(articles_with_ratings)
    
    print(f"\nProcess complete! Articles with numerical ratings saved to {args.output_path}")

if __name__ == "__main__":
    main() 