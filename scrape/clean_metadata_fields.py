#!/usr/bin/env python3
import json
import os

def main():
    input_file = 'best_articles/top5_per_source_cleaned.json'
    output_file = 'best_articles/top5_per_source_final.json'
    
    print(f"Reading articles from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    print(f"Loaded {len(articles)} articles")
    
    # Fields to remove from each article
    fields_to_remove = [
        'source_rating_value',
        'ai_political_leaning',
        'match_with_source',
        'ai_political_rating'  # Remove existing ratings so we can re-rate them
    ]
    
    # Remove the specified fields from each article
    modified_count = 0
    for article in articles:
        modified = False
        for field in fields_to_remove:
            if field in article:
                del article[field]
                modified = True
        
        if modified:
            modified_count += 1
    
    print(f"Removed metadata fields from {modified_count} articles")
    
    # Save the cleaned dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4)
    
    print(f"Saved cleaned dataset to {output_file}")
    
    # Show sample of first article
    if articles:
        print("\nSample first article:")
        print(f"Title: {articles[0].get('title', 'Unknown')}")
        print(f"Source: {articles[0].get('source_outlet', 'Unknown')}")
        print(f"Remaining fields: {', '.join(sorted(articles[0].keys()))}")

if __name__ == "__main__":
    main() 