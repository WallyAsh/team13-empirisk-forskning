#!/usr/bin/env python3
import json
import os
from collections import defaultdict
import heapq

def main():
    print("Loading articles from JSON file...")
    try:
        with open('models/deepseek/deepseek_articles_updated.json', 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    print(f"Loaded {len(articles)} articles")
    
    # Count articles by source outlet
    source_counts = defaultdict(int)
    for article in articles:
        if 'source_outlet' in article:
            source_counts[article['source_outlet']] += 1
    
    # Find sources with 5+ articles
    qualified_sources = {source: count for source, count in source_counts.items() if count >= 5}
    print(f"Found {len(qualified_sources)} source outlets with 5+ articles")
    
    # For each qualified source, find the 5 articles with the most text content
    best_articles_by_source = {}
    for source in qualified_sources:
        # Use min heap to keep track of 5 longest articles
        top_articles = []
        for i, article in enumerate(articles):
            if article.get('source_outlet') == source:
                text_length = len(article.get('full_text', ''))
                
                # If we have fewer than 5 articles, add this one
                if len(top_articles) < 5:
                    heapq.heappush(top_articles, (text_length, i))
                # Otherwise, replace the shortest if this one is longer
                elif text_length > top_articles[0][0]:
                    heapq.heappop(top_articles)
                    heapq.heappush(top_articles, (text_length, i))
        
        # Convert to list of articles, sorted by length (longest first)
        best_articles_by_source[source] = [articles[idx] for _, idx in sorted(top_articles, reverse=True)]
        print(f"Selected {len(best_articles_by_source[source])} articles for {source}")
    
    # Create the final dataset with top 5 articles from each source
    best_articles = []
    for source, articles_list in best_articles_by_source.items():
        best_articles.extend(articles_list)
    
    print(f"Total articles in best dataset: {len(best_articles)}")
    
    # Count sources by rating and articles by political leaning
    source_ratings = defaultdict(int)
    article_leanings = defaultdict(int)
    source_ratings_map = {}  # Store source outlet to rating mapping
    
    # First collect the source ratings
    for article in best_articles:
        source = article.get('source_outlet', '')
        rating = article.get('source_rating', 'Unknown')
        source_ratings_map[source] = rating
        
        # Count article political leanings
        ai_leaning = article.get('ai_political_leaning', 'Unknown')
        article_leanings[ai_leaning] += 1
    
    # Count source outlets by their rating
    for source, rating in source_ratings_map.items():
        source_ratings[rating] += 1
    
    # Save the dataset
    output_dir = 'best_articles'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'top5_per_source.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(best_articles, f, indent=4)
    
    print(f"Saved dataset to {output_file}")
    
    # Create a summary file with counts
    summary = {
        'total_articles': len(best_articles),
        'sources': {source: len(articles_list) for source, articles_list in best_articles_by_source.items()},
        'source_ratings': dict(source_ratings),
        'article_political_leanings': dict(article_leanings)
    }
    
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Saved summary to {summary_file}")

if __name__ == "__main__":
    main() 