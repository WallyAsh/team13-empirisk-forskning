#!/usr/bin/env python3
"""
Script to create a high-quality subset of articles for analysis.

This script:
1. Identifies news outlets with 5+ articles in the dataset
2. Selects 5 articles from each major outlet
3. Allows for manual review to ensure quality
4. Creates a clean subset for final analysis and classification
"""

import json
import pandas as pd
import os
import random
import sys
from pathlib import Path

# Default file paths
DEFAULT_ARTICLES_PATH = '../models/deepseek/deepseek_articles.json'
DEFAULT_OUTPUT_DIR = 'analysis_subset'
DEFAULT_SUBSET_PATH = os.path.join(DEFAULT_OUTPUT_DIR, 'selected_articles.json')
DEFAULT_FINAL_SUBSET_PATH = os.path.join(DEFAULT_OUTPUT_DIR, 'final_subset.json')

def load_articles(file_path=DEFAULT_ARTICLES_PATH):
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

def get_major_outlets(articles, min_articles=5):
    """Identify news outlets with at least min_articles articles"""
    outlet_counts = {}
    for article in articles:
        outlet = article.get('source_outlet')
        if outlet:
            outlet_counts[outlet] = outlet_counts.get(outlet, 0) + 1
    
    # Filter to outlets with min_articles or more
    major_outlets = {outlet: count for outlet, count in outlet_counts.items() 
                     if count >= min_articles}
    
    return major_outlets

def evaluate_article_quality(article):
    """Assign a quality score to an article based on various factors"""
    # Initialize quality score
    quality_score = 0
    full_text = article.get('full_text', '')
    
    # Check text length (longer articles tend to be more substantial)
    if full_text:
        text_length = len(full_text)
        quality_score += min(text_length / 1000, 10)  # Max 10 points for length
    else:
        return -100  # Severely penalize missing text
    
    # Check if the article has expected fields
    if 'title' in article and article['title']:
        quality_score += 2
    
    if 'source_rating' in article and article['source_rating']:
        quality_score += 1
    
    # Penalize "Not available" text
    if full_text == "Not available..." or full_text == "Not available":
        quality_score -= 100
    
    # Penalize very short articles or snippets
    if text_length < 500:
        quality_score -= 5
    
    return quality_score

def select_articles_from_outlets(articles, major_outlets, articles_per_outlet=5):
    """Select the best articles_per_outlet articles from each major outlet"""
    selected_articles = []
    outlet_articles = {}
    
    # Group articles by outlet
    for article in articles:
        outlet = article.get('source_outlet')
        if outlet in major_outlets:
            if outlet not in outlet_articles:
                outlet_articles[outlet] = []
            outlet_articles[outlet].append(article)
    
    # Select best articles from each outlet
    for outlet, outlet_article_list in outlet_articles.items():
        # Score articles
        scored_articles = [(article, evaluate_article_quality(article)) 
                          for article in outlet_article_list]
        
        # Sort by quality score (descending)
        scored_articles.sort(key=lambda x: x[1], reverse=True)
        
        # Select top articles
        top_articles = [article for article, score in scored_articles[:articles_per_outlet]]
        selected_articles.extend(top_articles)
        
        print(f"Selected {len(top_articles)} articles from {outlet}")
    
    return selected_articles

def manual_review_articles(articles):
    """Allow manual review and filtering of articles"""
    # Create CSV for easier review
    review_csv_path = os.path.join(DEFAULT_OUTPUT_DIR, 'review_articles.csv')
    df = pd.DataFrame([{
        'id': i,
        'outlet': article.get('source_outlet', ''),
        'title': article.get('title', ''),
        'source_rating': article.get('source_rating', ''),
        'ai_rating': article.get('ai_political_leaning', ''),
        'text_length': len(article.get('full_text', '')),
        'include': 'YES'  # Default to including all articles
    } for i, article in enumerate(articles)])
    
    df.to_csv(review_csv_path, index=False)
    
    print("\nManual review process:")
    print(f"1. A CSV file has been created at: {review_csv_path}")
    print("2. Open this file in Excel or your preferred spreadsheet application")
    print("3. Review the articles and change the 'include' column to 'NO' for any articles you want to exclude")
    print("4. Save the CSV file")
    print("5. Run this script again with the --apply-review flag to apply your changes")

def apply_manual_review(articles):
    """Apply manual review changes from the CSV file"""
    review_csv_path = os.path.join(DEFAULT_OUTPUT_DIR, 'review_articles.csv')
    
    if not os.path.exists(review_csv_path):
        print(f"Error: Review file {review_csv_path} not found.")
        return articles
    
    try:
        df = pd.read_csv(review_csv_path)
        included_ids = df[df['include'].str.upper() == 'YES']['id'].tolist()
        
        # Filter articles based on the review
        filtered_articles = [article for i, article in enumerate(articles) if i in included_ids]
        
        print(f"Applied manual review: Kept {len(filtered_articles)} out of {len(articles)} articles")
        return filtered_articles
    except Exception as e:
        print(f"Error applying manual review: {e}")
        return articles

def show_dataset_stats(articles):
    """Show statistics about the selected dataset"""
    # Count by outlet
    outlet_counts = {}
    for article in articles:
        outlet = article.get('source_outlet', 'Unknown')
        outlet_counts[outlet] = outlet_counts.get(outlet, 0) + 1
    
    # Count by source rating
    rating_counts = {}
    for article in articles:
        rating = article.get('source_rating', 'Unknown')
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
    
    print("\nDataset Statistics:")
    print("-" * 40)
    print(f"Total articles: {len(articles)}")
    
    print("\nArticles by outlet:")
    for outlet, count in sorted(outlet_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {outlet}: {count}")
    
    print("\nArticles by source rating:")
    for rating, count in sorted(rating_counts.items()):
        print(f"  {rating}: {count} ({count/len(articles)*100:.1f}%)")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a high-quality article subset for analysis")
    parser.add_argument('--articles-path', default=DEFAULT_ARTICLES_PATH,
                       help=f'Path to the articles JSON file (default: {DEFAULT_ARTICLES_PATH})')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR,
                       help=f'Directory to save output files (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--articles-per-outlet', type=int, default=5,
                       help='Number of articles to select per outlet (default: 5)')
    parser.add_argument('--apply-review', action='store_true',
                       help='Apply manual review changes from the CSV file')
    args = parser.parse_args()
    
    # Update paths based on output directory
    subset_path = os.path.join(args.output_dir, 'selected_articles.json')
    final_subset_path = os.path.join(args.output_dir, 'final_subset.json')
    
    print(f"Loading articles from {args.articles_path}...")
    articles = load_articles(args.articles_path)
    
    if not articles:
        print("No articles found. Exiting.")
        return
    
    print(f"Loaded {len(articles)} articles")
    
    # If applying review, load the previously selected subset
    if args.apply_review:
        if os.path.exists(subset_path):
            articles = load_articles(subset_path)
            filtered_articles = apply_manual_review(articles)
            save_articles(filtered_articles, final_subset_path)
            show_dataset_stats(filtered_articles)
        else:
            print(f"Error: Selected articles file {subset_path} not found.")
    else:
        # Select articles based on outlet
        major_outlets = get_major_outlets(articles, min_articles=args.articles_per_outlet)
        print(f"Found {len(major_outlets)} outlets with at least {args.articles_per_outlet} articles")
        
        for outlet, count in sorted(major_outlets.items(), key=lambda x: x[1], reverse=True):
            print(f"  {outlet}: {count} articles")
        
        # Select the best articles from each outlet
        selected_articles = select_articles_from_outlets(
            articles, major_outlets, args.articles_per_outlet)
        
        # Save selected articles for review
        save_articles(selected_articles, subset_path)
        show_dataset_stats(selected_articles)
        
        # Provide instructions for manual review
        manual_review_articles(selected_articles)

if __name__ == "__main__":
    main() 