#!/usr/bin/env python3
"""
Script to count the total number of unique news sources in articles_base.json and deepseek_articles.json
"""

import json
import pandas as pd
import os

def analyze_sources(file_path, file_name):
    """Analyze source information in the specified JSON file"""
    print(f"\n{'='*80}")
    print(f"Analyzing sources in {file_name}...")
    print(f"{'='*80}")
    
    # Load articles
    with open(file_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(articles)
    
    # Count total articles
    print(f"Total articles: {len(df)}")
    
    # Check for source_outlet field
    if 'source_outlet' in df.columns:
        sources = df['source_outlet'].dropna().unique()
        source_counts = df['source_outlet'].value_counts()
        
        print(f"\nTotal unique news sources: {len(sources)}")
        print("\nTop 20 sources by article count:")
        print("-" * 40)
        for source, count in source_counts.head(20).items():
            print(f"{source}: {count} articles")
        
        # Count sources with at least one, five, and ten articles
        sources_min_1 = sum(source_counts >= 1)
        sources_min_5 = sum(source_counts >= 5)
        sources_min_10 = sum(source_counts >= 10)
        
        print("\nSource distribution:")
        print(f"Sources with at least 1 article: {sources_min_1}")
        print(f"Sources with at least 5 articles: {sources_min_5}")
        print(f"Sources with at least 10 articles: {sources_min_10}")
    else:
        print("No 'source_outlet' column found in the data")
        
    # Get source rating distribution if available
    if 'source_rating' in df.columns:
        rating_counts = df['source_rating'].value_counts()
        print("\nSource rating distribution:")
        print("-" * 40)
        for rating, count in rating_counts.items():
            print(f"{rating}: {count} articles ({count/len(df)*100:.1f}%)")
    
    # Get AI political leaning distribution if available
    if 'ai_political_leaning' in df.columns:
        ai_counts = df['ai_political_leaning'].value_counts()
        print("\nAI political leaning distribution:")
        print("-" * 40)
        for rating, count in ai_counts.items():
            print(f"{rating}: {count} articles ({count/len(df)*100:.1f}%)")
    
    return df

def main():
    """Main function"""
    print("Starting source analysis...")
    
    # Analyze articles_base.json
    articles_base_df = analyze_sources('data/articles_base.json', 'articles_base.json')
    
    # Analyze deepseek_articles.json
    deepseek_df = analyze_sources('models/deepseek/deepseek_articles.json', 'deepseek_articles.json')
    
    # Compare the two datasets
    print("\n" + "="*80)
    print("Comparison between the two datasets")
    print("="*80)
    
    print(f"Total articles in articles_base.json: {len(articles_base_df)}")
    print(f"Total articles in deepseek_articles.json: {len(deepseek_df)}")
    
    # Calculate the overlap between the two datasets
    if 'title' in articles_base_df.columns and 'title' in deepseek_df.columns:
        base_titles = set(articles_base_df['title'])
        deepseek_titles = set(deepseek_df['title'])
        overlap = base_titles.intersection(deepseek_titles)
        
        print(f"\nArticles present in both datasets: {len(overlap)}")
        print(f"Articles unique to articles_base.json: {len(base_titles - deepseek_titles)}")
        print(f"Articles unique to deepseek_articles.json: {len(deepseek_titles - base_titles)}")

if __name__ == "__main__":
    main() 