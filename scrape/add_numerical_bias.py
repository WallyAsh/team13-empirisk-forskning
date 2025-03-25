#!/usr/bin/env python3
"""
Add numerical bias values to article databases.
Converts political leanings to numerical values:
- Left: -2
- Lean Left: -1
- Center: 0
- Lean Right: 1
- Right: 2
"""

import json
import pandas as pd
import os

# Define the bias mapping
BIAS_MAPPING = {
    'Left': -2,
    'Lean Left': -1,
    'Center': 0,
    'Lean Right': 1,
    'Right': 2,
    'Mixed': 0,  # Treat mixed as center
    'Not rated': None,  # Keep unknown as None
    'No content': None,  # Keep no content as None
    'left': -2,  # Handle lowercase variants
    'left-leaning': -1,
    'center': 0,
    'right-leaning': 1,
    'right': 2
}

def add_numerical_bias(file_path):
    """
    Add numerical bias values to a JSON file.
    """
    print(f"Processing {file_path}...")
    
    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Add numerical bias values
    for article in data:
        # Add source rating numerical value
        source_rating = article.get('source_rating', 'Not rated')
        article['source_rating_value'] = BIAS_MAPPING.get(source_rating)
        
        # Add AI classification numerical value if it exists
        if 'ai_political_leaning' in article:
            ai_rating = article['ai_political_leaning']
            article['ai_rating_value'] = BIAS_MAPPING.get(ai_rating)
            
            # Calculate the difference between AI and source ratings
            if article['source_rating_value'] is not None and article['ai_rating_value'] is not None:
                article['rating_difference'] = article['ai_rating_value'] - article['source_rating_value']
            else:
                article['rating_difference'] = None
    
    # Save back to JSON
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    # Also save as CSV
    csv_path = file_path.replace('.json', '.csv')
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    print(f"Updated {file_path} and {csv_path}")
    
    # Print some statistics
    print("\nStatistics:")
    print(f"Total articles: {len(data)}")
    
    # Source rating distribution
    source_dist = pd.DataFrame(data)['source_rating'].value_counts()
    print("\nSource Rating Distribution:")
    for rating, count in source_dist.items():
        print(f"{rating}: {count} ({count/len(data)*100:.1f}%)")
    
    # AI rating distribution (if exists)
    if 'ai_political_leaning' in data[0]:
        ai_dist = pd.DataFrame(data)['ai_political_leaning'].value_counts()
        print("\nAI Rating Distribution:")
        for rating, count in ai_dist.items():
            print(f"{rating}: {count} ({count/len(data)*100:.1f}%)")
    
    # Rating difference statistics (if applicable)
    if 'rating_difference' in data[0]:
        diffs = [d['rating_difference'] for d in data if d['rating_difference'] is not None]
        if diffs:
            print("\nRating Difference Statistics:")
            print(f"Mean difference: {sum(diffs)/len(diffs):.2f}")
            print(f"Max difference: {max(diffs)}")
            print(f"Min difference: {min(diffs)}")
            print(f"Zero differences: {diffs.count(0)} ({diffs.count(0)/len(diffs)*100:.1f}%)")

def main():
    """Main function to add numerical bias values to all relevant files."""
    # Files to process
    files = [
        'data/articles_base.json',
        'models/deepseek/deepseek_articles.json'
    ]
    
    print("Starting to add numerical bias values...")
    
    for file_path in files:
        if os.path.exists(file_path):
            add_numerical_bias(file_path)
        else:
            print(f"Warning: {file_path} not found")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 