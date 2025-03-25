#!/usr/bin/env python3
"""
Script to scrape numerical bias ratings from AllSides source pages.

This script:
1. Loads articles from the DeepSeek database
2. Extracts unique source URLs
3. Visits each AllSides page to scrape the numerical bias rating
4. Updates the database with the precise numerical ratings
"""

import json
import requests
import time
import random
import os
import sys
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import pandas as pd
from tqdm import tqdm
import cloudscraper

# Default file paths
DEFAULT_ARTICLES_PATH = 'models/deepseek/deepseek_articles.json'
DEFAULT_OUTPUT_PATH = 'models/deepseek/deepseek_articles_updated.json'
DEFAULT_CACHE_PATH = 'data/allsides_ratings_cache.json'

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

def load_cache(cache_path):
    """Load cached ratings from JSON file"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    return {}

def save_cache(cache, cache_path):
    """Save cached ratings to JSON file"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(cache)} ratings to cache")

def extract_unique_sources(articles):
    """Extract unique source URLs from articles"""
    unique_sources = {}
    for article in articles:
        source_url = article.get('source_url')
        if source_url and 'allsides.com' in source_url and source_url not in unique_sources:
            unique_sources[source_url] = article.get('source_outlet', 'Unknown')
    
    print(f"Found {len(unique_sources)} unique AllSides source URLs")
    return unique_sources

def extract_numerical_rating(url, cache):
    """Extract numerical bias rating from AllSides source page"""
    # Check cache first
    if url in cache:
        return cache[url]
    
    try:
        # Setup headers and scraper like in update_database.py
        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com",
            "Upgrade-Insecure-Requests": "1",
            "Connection": "keep-alive",
            "DNT": "1"
        }
        
        # Create a cloudscraper instance
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url, headers=headers, timeout=15)
        
        # Check response status
        if response.status_code != 200:
            print(f"Error: Unable to retrieve page (status code {response.status_code})")
            return None
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find the numerical rating
        rating_div = soup.select_one('div.numerical-bias-rating')
        if rating_div:
            try:
                rating = float(rating_div.text.strip())
                # Cache the result
                cache[url] = rating
                return rating
            except ValueError:
                print(f"Could not parse rating from {rating_div.text} for {url}")
        else:
            print(f"No numerical rating found for {url}")
        
        return None
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def get_random_user_agent():
    """Get a random user agent to avoid detection."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.188",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
    ]
    return random.choice(user_agents)

def update_articles_with_ratings(articles, ratings):
    """Update articles with scraped numerical ratings"""
    updated_count = 0
    for article in articles:
        source_url = article.get('source_url')
        if source_url in ratings and ratings[source_url] is not None:
            article['source_rating_value_precise'] = ratings[source_url]
            updated_count += 1
    
    print(f"Updated {updated_count} articles with precise numerical ratings")
    return articles

def display_rating_stats(articles):
    """Display statistics about the scraped ratings"""
    # Count articles with precise ratings
    precise_count = sum(1 for article in articles if 'source_rating_value_precise' in article)
    
    # Get all unique precise ratings
    precise_ratings = set()
    rating_by_category = {}
    
    for article in articles:
        if 'source_rating_value_precise' in article:
            rating = article['source_rating_value_precise']
            precise_ratings.add(rating)
            
            category = article.get('source_rating', 'Unknown')
            if category not in rating_by_category:
                rating_by_category[category] = []
            rating_by_category[category].append(rating)
    
    # Display results
    print("\nRating Statistics:")
    print(f"Articles with precise ratings: {precise_count}/{len(articles)} ({precise_count/len(articles)*100:.1f}%)")
    print(f"Unique precise ratings found: {len(precise_ratings)}")
    
    print("\nRating ranges by category:")
    for category, ratings in sorted(rating_by_category.items()):
        if ratings:
            min_val = min(ratings)
            max_val = max(ratings)
            avg_val = sum(ratings) / len(ratings)
            print(f"  {category}: {min_val} to {max_val} (avg: {avg_val:.2f}, count: {len(ratings)})")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape numerical bias ratings from AllSides")
    parser.add_argument('--articles-path', default=DEFAULT_ARTICLES_PATH,
                       help=f'Path to the articles JSON file (default: {DEFAULT_ARTICLES_PATH})')
    parser.add_argument('--output-path', default=DEFAULT_OUTPUT_PATH,
                       help=f'Path to save the updated articles (default: {DEFAULT_OUTPUT_PATH})')
    parser.add_argument('--cache-path', default=DEFAULT_CACHE_PATH,
                       help=f'Path to the ratings cache file (default: {DEFAULT_CACHE_PATH})')
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh all ratings, ignoring cache')
    parser.add_argument('--delay-min', type=float, default=3.0,
                       help='Minimum delay between requests in seconds (default: 3.0)')
    parser.add_argument('--delay-max', type=float, default=7.0,
                       help='Maximum delay between requests in seconds (default: 7.0)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum number of retry attempts for failed requests (default: 3)')
    args = parser.parse_args()
    
    # Load articles
    print(f"Loading articles from {args.articles_path}...")
    articles = load_articles(args.articles_path)
    
    if not articles:
        print("No articles found. Exiting.")
        return
    
    print(f"Loaded {len(articles)} articles")
    
    # Check if cloudscraper is installed
    try:
        import cloudscraper
    except ImportError:
        print("Error: cloudscraper package is required. Please install it with:")
        print("pip install cloudscraper")
        return
    
    # Load cache
    cache = {} if args.force_refresh else load_cache(args.cache_path)
    print(f"Loaded {len(cache)} cached ratings")
    
    # Extract unique source URLs
    unique_sources = extract_unique_sources(articles)
    
    # Scrape numerical ratings
    ratings = {}
    print("\nScraping numerical ratings...")
    for url, source_name in tqdm(unique_sources.items(), desc="Scraping"):
        # Check if we already have this rating in the cache
        if url in cache and not args.force_refresh:
            ratings[url] = cache[url]
            continue
        
        # Try to scrape with retries
        rating = None
        for attempt in range(args.max_retries):
            try:
                rating = extract_numerical_rating(url, cache)
                if rating is not None:
                    break
                print(f"Retry {attempt+1}/{args.max_retries} for {url}")
            except Exception as e:
                print(f"Error on attempt {attempt+1}/{args.max_retries}: {e}")
                time.sleep(random.uniform(args.delay_min, args.delay_max))
        
        if rating is not None:
            ratings[url] = rating
            print(f"Successfully found rating {rating} for {source_name}")
        else:
            print(f"Failed to get rating for {source_name} after {args.max_retries} attempts")
        
        # Add a delay to avoid overloading the server
        delay = random.uniform(args.delay_min, args.delay_max)
        print(f"Waiting {delay:.1f} seconds before next request...")
        time.sleep(delay)
    
    # Save updated cache
    save_cache(cache, args.cache_path)
    
    # Update articles with numerical ratings
    updated_articles = update_articles_with_ratings(articles, ratings)
    
    # Display statistics
    display_rating_stats(updated_articles)
    
    # Save updated articles
    save_articles(updated_articles, args.output_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 