#!/usr/bin/env python3
"""
Script to manually review article texts and mark them as sufficient or insufficient.
"""

import json
import pandas as pd
import os
from datetime import datetime

def load_articles():
    """Load articles from the DeepSeek JSON file."""
    with open('models/deepseek/deepseek_articles.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def save_review_results(articles, review_file):
    """Save the review results to a JSON file."""
    with open(review_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)

def review_articles():
    """Review article texts and mark them as sufficient or insufficient."""
    # Load articles
    articles = load_articles()
    
    # Filter out articles with unavailable text
    articles = [a for a in articles if a['full_text'] != "Not available..."]
    
    # Create review file path
    review_file = 'models/deepseek/article_text_reviews.json'
    
    # Review each article
    for i, article in enumerate(articles, 1):
        print(f"\nArticle {i}/{len(articles)}")
        print("=" * 80)
        print(f"Title: {article.get('title', 'No title')}")
        print(f"Source: {article.get('source', 'Unknown source')}")
        print(f"Source Rating: {article.get('source_rating', 'Not rated')}")
        print(f"AI Rating: {article.get('ai_political_leaning', 'Not classified')}")
        print("\nFull Text:")
        print("-" * 80)
        print(article['full_text'])
        print("-" * 80)
        
        while True:
            response = input("\nIs this text sufficient for analysis? (y/n/s to skip): ").lower()
            if response in ['y', 'n', 's']:
                break
            print("Please enter 'y' for yes, 'n' for no, or 's' to skip")
        
        if response == 's':
            continue
            
        article['text_sufficient'] = response == 'y'
        if not article['text_sufficient']:
            notes = input("Please enter notes about why the text is insufficient: ")
            article['review_notes'] = notes
        
        # Save progress after each review
        save_review_results(articles, review_file)
        
        # Ask if user wants to continue
        if i < len(articles):
            continue_review = input("\nContinue reviewing? (y/n): ").lower()
            if continue_review != 'y':
                break
    
    # Print summary
    reviewed = [a for a in articles if 'text_sufficient' in a]
    sufficient = [a for a in reviewed if a['text_sufficient']]
    
    print("\nReview Summary:")
    print(f"Total articles reviewed: {len(reviewed)}")
    print(f"Articles with sufficient text: {len(sufficient)}")
    print(f"Articles with insufficient text: {len(reviewed) - len(sufficient)}")
    print(f"Remaining articles to review: {len(articles) - len(reviewed)}")

def main():
    """Main function to run the review process."""
    print("Starting article text review process...")
    review_articles()
    print("\nReview process complete!")

if __name__ == "__main__":
    main() 