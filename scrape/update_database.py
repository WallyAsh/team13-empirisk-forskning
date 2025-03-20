#!/usr/bin/env python3
"""
Database Updater - Central script for scraping and managing the article database

This script handles:
1. Scraping articles from AllSides
2. Extracting full text from original sources
3. Maintaining the central article database
4. No classification (that's handled by model-specific processors)
"""

import json
import pandas as pd
import time
import random
import requests
import os
import sys
from bs4 import BeautifulSoup
from newspaper import Article
import cloudscraper
from tqdm import tqdm
from urllib.parse import urlparse

# File paths
DATABASE_JSON_PATH = "data/articles_base.json"
DATABASE_CSV_PATH = "data/articles_base.csv"
DEFAULT_URL = "https://www.allsides.com/unbiased-balanced-news"

#--------------------------------
# Web Scraping Functions
#--------------------------------

def get_random_user_agent():
    """Get a random user agent to avoid detection."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:88.0) Gecko/20100101 Firefox/88.0"
    ]
    return random.choice(user_agents)

def extract_article_text(url):
    """
    Extract full text from an article URL using multiple methods.
    Returns the extracted text or None if all methods fail.
    """
    if not url or url == "Not found":
        return "Not available..."
    
    # Method 1: Using newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        
        if text and len(text.strip()) > 200:  # Ensure we have meaningful content
            return text
    except Exception as e:
        print(f"newspaper3k extraction failed: {e}")
    
    # Method 2: Using requests + BeautifulSoup
    try:
        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/",
            "DNT": "1"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style elements and comments
            for element in soup(["script", "style"]):
                element.decompose()
            
            # Common article body selectors
            selectors = [
                "article", "div.article-body", "div.story-body", 
                "div.article-content", "div.content-body", "div.story-content",
                "div.post-content", "div.entry-content", "main#main",
                "div.main-content", "div.article", "div.post", "div.story"
            ]
            
            article_text = ""
            
            # Try different selectors
            for selector in selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    paragraphs = main_content.find_all('p')
                    if paragraphs:
                        article_text = '\n\n'.join([p.get_text().strip() for p in paragraphs])
                        break
            
            # If no content found with selectors, get all paragraphs
            if not article_text:
                paragraphs = soup.find_all('p')
                article_text = '\n\n'.join([p.get_text().strip() for p in paragraphs])
            
            if article_text and len(article_text.strip()) > 200:
                return article_text
    except Exception as e:
        print(f"BeautifulSoup extraction failed: {e}")
    
    # Method 3: Using cloudscraper (for sites with anti-bot measures)
    try:
        scraper = cloudscraper.create_scraper()
        response = scraper.get(url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script, style elements
            for element in soup(["script", "style"]):
                element.decompose()
            
            # Try with same selectors as in Method 2
            selectors = [
                "article", "div.article-body", "div.story-body", 
                "div.article-content", "div.content-body", "div.story-content",
                "div.post-content", "div.entry-content", "main#main",
                "div.main-content", "div.article", "div.post", "div.story"
            ]
            
            article_text = ""
            
            for selector in selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    paragraphs = main_content.find_all('p')
                    if paragraphs:
                        article_text = '\n\n'.join([p.get_text().strip() for p in paragraphs])
                        break
            
            if not article_text:
                paragraphs = soup.find_all('p')
                article_text = '\n\n'.join([p.get_text().strip() for p in paragraphs])
            
            if article_text and len(article_text.strip()) > 200:
                return article_text
    except Exception as e:
        print(f"Cloudscraper extraction failed: {e}")
    
    # If all methods fail
    return "Not available..."

def get_original_source(article_url, scraper, headers):
    """Get the original source URL from an AllSides article page."""
    try:
        response = scraper.get(article_url, headers=headers)
        if response.status_code != 200:
            return "Not found"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the "Read It At" link
        read_it_at_div = soup.find('div', class_='read-more-story')
        if read_it_at_div:
            link = read_it_at_div.find('a')
            if link:
                return link.get('href')
        
        # Try alternative methods to find original source
        # Look for any link that might point to the original article
        possible_containers = [
            soup.select('.news-source a'),
            soup.select('.field-name-field-story-url a'),
            soup.select('.field-name-body a')
        ]
        
        for container in possible_containers:
            if container:
                for link in container:
                    href = link.get('href')
                    if href and not href.startswith(('https://www.allsides.com', '/news')):
                        return href
        
        return "Not found"
    except Exception as e:
        print(f"Error getting original source: {e}")
        return "Not found"

def scrape_allsides_page(url):
    """Scrape articles from an AllSides page."""
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com",
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive",
        "DNT": "1"
    }
    
    scraper = cloudscraper.create_scraper()
    response = scraper.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error: Unable to retrieve page (status code {response.status_code})")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Select all news-item blocks inside the news-trio container
    news_items = soup.select("div.news-trio div.news-item")
    scraped_articles = []
    
    for item in news_items:
        link_tag = item.find('a')
        if link_tag is None:
            continue

        link = link_tag.get('href', 'No link found')
        if link.startswith('/'):
            link = "https://www.allsides.com" + link

        title_div = link_tag.find('div', class_='news-title')
        title = title_div.get_text(strip=True) if title_div else link_tag.get_text(strip=True)
        
        source_tag = item.find('a', class_='source-area')
        if source_tag:
            outlet_tag = source_tag.find('div', class_='news-source')
            outlet = outlet_tag.get_text(strip=True) if outlet_tag else source_tag.get_text(strip=True)
            source_url = source_tag.get('href', 'No source URL found')
            
            img_tag = source_tag.find('img')
            if img_tag:
                alt_text = img_tag.get('alt', '')
                if "Rating:" in alt_text:
                    rating = alt_text.split("Rating:")[-1].strip()
                else:
                    rating = "Not rated"
            else:
                rating = "Not rated"
        else:
            outlet = "No source found"
            source_url = "No source URL found"
            rating = "Not rated"
        
        # Get the original source URL
        original_source = get_original_source(link, scraper, headers)
        
        # Create article object
        article = {
            "title": title,
            "link": link,
            "source_outlet": outlet,
            "source_url": source_url,
            "source_rating": rating,
            "original_source": original_source,
            "full_text": "Not available..."  # Will be filled in later
        }
        
        scraped_articles.append(article)
    
    return scraped_articles

#--------------------------------
# Data Management Functions
#--------------------------------

def load_existing_articles(file_path):
    """
    Load existing articles from a JSON file if it exists.
    Returns an empty list if the file doesn't exist.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing articles: {e}")
            return []
    return []

def merge_articles(existing_articles, new_articles):
    """
    Merge new articles with existing ones, avoiding duplicates.
    Uses article link as the unique identifier.
    """
    # Create a dict of existing articles for faster lookup
    existing_links = {article['link']: article for article in existing_articles}
    
    # Add all new articles that aren't already in the existing set
    merged_articles = existing_articles.copy()
    for article in new_articles:
        if article['link'] not in existing_links:
            merged_articles.append(article)
    
    return merged_articles

def save_articles(articles, json_path, csv_path):
    """Save articles to JSON and CSV files."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    
    # Save to JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
    
    # Save to CSV
    df = pd.DataFrame(articles)
    df.to_csv(csv_path, index=False)
    
    print(f"JSON file saved as {json_path}")
    print(f"CSV file saved as {csv_path}")

def print_dataset_stats(articles):
    """Print statistics about the article dataset."""
    # Count articles with full text
    total_with_text = sum(1 for a in articles if a.get('full_text') and a['full_text'] != "Not available...")
    
    # Count articles by source rating
    ratings = {}
    for article in articles:
        rating = article.get('source_rating', 'Unknown')
        ratings[rating] = ratings.get(rating, 0) + 1
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total articles: {len(articles)}")
    print(f"Articles with full text: {total_with_text} ({total_with_text/len(articles)*100:.1f}%)")
    
    print("\nSource Ratings Distribution:")
    for rating, count in sorted(ratings.items()):
        percentage = (count / len(articles) * 100)
        print(f"{rating}: {count} ({percentage:.1f}%)")

#--------------------------------
# Main Workflows
#--------------------------------

def initial_setup(url=DEFAULT_URL, json_path=DATABASE_JSON_PATH, csv_path=DATABASE_CSV_PATH):
    """
    Initial setup workflow:
    1. Scrape articles from AllSides
    2. Extract full text for each article
    3. Save the dataset
    """
    print(f"Starting initial setup from {url}")
    
    # Scrape articles
    print(f"Scraping articles from {url}...")
    scraped_articles = scrape_allsides_page(url)
    
    # Extract full text for each article
    print("Extracting full text for articles...")
    for article in tqdm(scraped_articles, desc="Extracting text"):
        if article.get('original_source') and article['original_source'] != "Not found":
            try:
                article['full_text'] = extract_article_text(article['original_source'])
                # Add a small delay to avoid rate limiting
                time.sleep(random.uniform(1.0, 2.0))
            except Exception as e:
                print(f"Error extracting text for {article['title']}: {e}")
                article['full_text'] = "Not available..."
    
    # Save results
    save_articles(scraped_articles, json_path, csv_path)
    
    # Print statistics
    print_dataset_stats(scraped_articles)
    
    print("\nInitial setup complete!")
    return scraped_articles

def update_articles(url=DEFAULT_URL, json_path=DATABASE_JSON_PATH, csv_path=DATABASE_CSV_PATH):
    """
    Update workflow:
    1. Load existing articles
    2. Scrape new articles from AllSides
    3. Extract full text only for new articles
    4. Save the updated dataset
    """
    # Load existing articles
    existing_articles = load_existing_articles(json_path)
    if not existing_articles:
        print(f"No existing articles found in {json_path}")
        print("Switching to initial setup mode...")
        return initial_setup(url, json_path, csv_path)
    
    print(f"Loaded {len(existing_articles)} existing articles from {json_path}")
    
    # Scrape new articles
    print(f"Checking for new articles from {url}...")
    new_articles = scrape_allsides_page(url)
    
    # Find truly new articles
    existing_links = {article['link'] for article in existing_articles}
    truly_new_articles = [a for a in new_articles if a['link'] not in existing_links]
    
    print(f"Found {len(truly_new_articles)} new articles")
    
    if truly_new_articles:
        # Extract text for new articles
        print("Extracting full text for new articles...")
        for article in tqdm(truly_new_articles, desc="Extracting text"):
            if not article.get('full_text') or article['full_text'] == "Not available...":
                if article.get('original_source') and article['original_source'] != "Not found":
                    try:
                        article['full_text'] = extract_article_text(article['original_source'])
                        # Add a small delay to avoid rate limiting
                        time.sleep(random.uniform(2.0, 4.0))
                    except Exception as e:
                        print(f"Error extracting text for {article['title']}: {e}")
                        article['full_text'] = "Not available..."
                else:
                    article['full_text'] = "Not available..."
        
        # Merge and save
        all_articles = merge_articles(existing_articles, truly_new_articles)
        save_articles(all_articles, json_path, csv_path)
        
        # Print details of new articles
        print("\nNew articles added:")
        for article in truly_new_articles:
            print(f"Title: {article['title']}")
            print(f"Source: {article['source_outlet']} ({article['source_rating']})")
            print(f"Full Text Preview: {article['full_text'][:100]}..." if article.get('full_text') and article['full_text'] != "Not available..." else "Not available...")
            print("-" * 40)
        
        # Print overall statistics 
        print_dataset_stats(all_articles)
    else:
        print("No new articles found. Dataset is already up to date.")
    
    print("\nUpdate complete!")

def extract_missing_text(json_path=DATABASE_JSON_PATH, csv_path=DATABASE_CSV_PATH):
    """
    Extract missing text from articles that don't have full text.
    """
    print(f"Starting extract_missing_text function with path: {json_path}")
    
    # Load existing articles
    existing_articles = load_existing_articles(json_path)
    if not existing_articles:
        print(f"No existing articles found in {json_path}")
        return
    
    print(f"Loaded {len(existing_articles)} articles from {json_path}")
    
    # Find articles without text
    articles_without_text = [article for article in existing_articles 
                           if 'full_text' not in article or not article['full_text'] 
                           or article['full_text'] == "Not available..."]
    
    print(f"Found {len(articles_without_text)} articles without text")
    
    # Debug: Print first few articles without text to verify
    if articles_without_text:
        print("Sample of articles without text:")
        for i, article in enumerate(articles_without_text[:3]):
            print(f"  {i+1}. Title: {article.get('title', 'No title')}")
            print(f"     Link: {article.get('link', 'No link')}")
            print(f"     Original source: {article.get('original_source', 'None')}")
    else:
        print("No articles without text found. Nothing to do.")
    
    if articles_without_text:
        # Extract text for articles without text
        print("Extracting missing text...")
        updated_count = 0
        
        for article in tqdm(articles_without_text, desc="Extracting text"):
            if article.get('original_source') and article['original_source'] != "Not found":
                try:
                    text = extract_article_text(article['original_source'])
                    if text and text != "Not available...":
                        # Update the article in the main list
                        for main_article in existing_articles:
                            if main_article['link'] == article['link']:
                                main_article['full_text'] = text
                                updated_count += 1
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(random.uniform(2.0, 4.0))
                except Exception as e:
                    print(f"Error extracting text for {article.get('title', 'Unknown title')}: {e}")
        
        if updated_count > 0:
            # Save updated dataset
            save_articles(existing_articles, json_path, csv_path)
            print(f"\nUpdated {updated_count} articles with missing text")
        else:
            print("\nNo articles were updated with missing text")
    else:
        print("No articles without text found. Dataset is complete.")

def extract_missing_and_notfound(json_path=DATABASE_JSON_PATH, csv_path=DATABASE_CSV_PATH):
    """
    Extract missing text AND retry finding original sources for articles where they weren't found.
    This combines the functionality of extract_missing_text and extract_notfound_articles.py.
    """
    print(f"Starting enhanced extraction process with path: {json_path}")
    
    # Load existing articles
    existing_articles = load_existing_articles(json_path)
    if not existing_articles:
        print(f"No existing articles found in {json_path}")
        return
    
    print(f"Loaded {len(existing_articles)} articles from {json_path}")
    
    # Process in two phases:
    # 1. First, retry finding original sources for "Not found" articles
    not_found_articles = [article for article in existing_articles if article.get('original_source') == "Not found"]
    print(f"Found {len(not_found_articles)} articles with 'Not found' original sources")
    
    source_found_count = 0
    updated_count = 0
    
    # Setup for source finding
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com",
        "DNT": "1"
    }
    scraper = cloudscraper.create_scraper()
    
    # Save progress periodically
    save_interval = 10
    last_save = 0
    
    # Phase 1: Find original sources
    if not_found_articles:
        print("Retrying original source detection...")
        for i, article in enumerate(tqdm(not_found_articles, desc="Finding sources")):
            try:
                # Use the improved source finding method
                original_source = get_original_source(article['link'], scraper, headers)
                if original_source != "Not found":
                    article['original_source'] = original_source
                    source_found_count += 1
                    
                    # Try to immediately extract text too
                    try:
                        full_text = extract_article_text(original_source)
                        if full_text != "Not available...":
                            article['full_text'] = full_text
                            updated_count += 1
                            print(f"Found and extracted: {article.get('title', 'Unknown')[:40]}...")
                    except Exception as text_err:
                        print(f"Error extracting text: {text_err}")
                
                # Save progress periodically
                if (i + 1) % save_interval == 0:
                    print(f"\nIntermediate save at {i+1}/{len(not_found_articles)} articles")
                    save_articles(existing_articles, json_path, csv_path)
                    last_save = i + 1
                
                # Small delay to avoid rate limiting
                time.sleep(random.uniform(1.0, 2.0))
            except Exception as e:
                print(f"Error finding source for {article.get('title', 'Unknown title')}: {e}")
                # Continue to next article despite error
                continue
    
    # Save if we found any sources and haven't saved recently
    if source_found_count > last_save:
        print(f"\nSaving after finding {source_found_count} original sources")
        save_articles(existing_articles, json_path, csv_path)
    
    print(f"Successfully found {source_found_count} original sources")
    print(f"Successfully extracted text for {updated_count} articles")
    
    # Phase 2: Now extract text for all remaining articles without text
    articles_without_text = [article for article in existing_articles 
                          if ('full_text' not in article or not article['full_text'] or article['full_text'] == "Not available...")
                          and article.get('original_source') != "Not found"]
    
    print(f"Found {len(articles_without_text)} remaining articles without text but with original sources")
    
    if articles_without_text:
        print("Extracting missing text...")
        more_updated_count = 0
        last_save = 0
        
        for i, article in enumerate(tqdm(articles_without_text, desc="Extracting text")):
            try:
                full_text = extract_article_text(article['original_source'])
                if full_text != "Not available...":
                    article['full_text'] = full_text
                    more_updated_count += 1
                
                # Save progress periodically
                if (i + 1) % save_interval == 0:
                    print(f"\nIntermediate save at {i+1}/{len(articles_without_text)} articles")
                    save_articles(existing_articles, json_path, csv_path)
                    last_save = i + 1
                
                # Add a small delay to avoid rate limiting
                time.sleep(random.uniform(1.0, 2.0))
            except Exception as e:
                print(f"Error extracting text from {article.get('original_source', 'Unknown URL')}: {e}")
                # Continue to next article despite error
                continue
        
        print(f"\nSuccessfully extracted text for additional {more_updated_count} articles")
    
    # Save the final updated dataset
    total_updated = updated_count + more_updated_count
    if source_found_count > 0 or total_updated > 0:
        save_articles(existing_articles, json_path, csv_path)
        print(f"Updated dataset saved to {json_path} and {csv_path}")
        print(f"Total updates: {source_found_count} new sources, {total_updated} articles with text")
    else:
        print("No updates were made to the dataset")
    
    # Print overall statistics
    print_dataset_stats(existing_articles)

#--------------------------------
# Main
#--------------------------------

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AllSides Article Database Manager - Scrape, extract text, and maintain article database')
    
    # Main modes
    parser.add_argument('--initial-setup', action='store_true',
                      help='Run full initial setup (scrape, extract text)')
    parser.add_argument('--update', action='store_true',
                      help='Update mode - only process new articles (default if no mode specified)')
    parser.add_argument('--extract-missing', action='store_true',
                      help='Extract missing text from articles that already exist')
    parser.add_argument('--enhanced-extract', action='store_true',
                      help='Enhanced extraction - both retry finding sources AND extract missing text')
    
    # Common options
    parser.add_argument('--url', default=DEFAULT_URL,
                      help=f'URL to scrape (default: {DEFAULT_URL})')
    parser.add_argument('--json-path', default=DATABASE_JSON_PATH,
                      help=f'Path to save/load JSON file (default: {DATABASE_JSON_PATH})')
    parser.add_argument('--csv-path', default=DATABASE_CSV_PATH,
                      help=f'Path to save CSV file (default: {DATABASE_CSV_PATH})')
    
    args = parser.parse_args()
    
    try:
        if args.initial_setup:
            initial_setup(args.url, args.json_path, args.csv_path)
        elif args.extract_missing:
            extract_missing_text(args.json_path, args.csv_path)
        elif args.enhanced_extract:
            extract_missing_and_notfound(args.json_path, args.csv_path)
        else:  # Default is update mode
            update_articles(args.url, args.json_path, args.csv_path)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}") 