#!/usr/bin/env python3
"""
News Processor - Scrape AllSides, extract article text, and classify political bias

Two main modes:
1. Initial setup: Scrape AllSides, extract full text, classify articles
2. Update mode: Check for new articles, extract their text, classify only new articles
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
from openai import OpenAI

# DeepSeek API configuration
DEEPSEEK_API_KEY = "sk-739ab05b7a6d4853bd6615e48387cd2f"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Configure the OpenAI client to use DeepSeek
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# Define the categories we're interested in
CATEGORIES = ["Left", "Lean Left", "Center", "Lean Right", "Right"]

# File paths
DEFAULT_JSON_PATH = "allsides_articles.json"
DEFAULT_CSV_PATH = "allsides_articles.csv"
DEFAULT_URL = "https://www.allsides.com/unbiased-balanced-news"

#--------------------------------
# Web Scraping Functions
#--------------------------------

def get_random_user_agent():
    """Return a random user agent to avoid detection."""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59"
    ]
    return random.choice(user_agents)

def extract_article_text(url):
    """Extract the full text from an article URL using newspaper3k."""
    if not url or url == "Not found":
        return "No URL provided"
    
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # Basic validation
        text = article.text.strip()
        if not text or len(text) < 100:
            return "Insufficient text extracted"
        
        return text
    except Exception as e:
        return f"Failed to extract article text: {str(e)}"

def get_original_source(article_url, scraper, headers):
    """Extract the original source URL from an AllSides article page."""
    try:
        response = scraper.get(article_url, headers=headers)
        if response.status_code != 200:
            return "Not found"
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the "Read It At" link
        read_it_at_div = soup.find('div', class_='read-it-at')
        if read_it_at_div:
            link = read_it_at_div.find('a')
            if link:
                return link.get('href')
        
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
        
        # Get the original source URL and extract article text
        original_source = get_original_source(link, scraper, headers)
        
        # Delay between requests to be nice to the server
        time.sleep(random.uniform(1.0, 2.0))
        
        article = {
            'title': title,
            'link': link,
            'source_outlet': outlet,
            'source_url': source_url,
            'source_rating': rating,
            'original_source': original_source
        }
        
        # Extract the full text if we have the original source
        if original_source and original_source != "Not found":
            article['full_text'] = extract_article_text(original_source)
        else:
            article['full_text'] = "Not available"
        
        scraped_articles.append(article)
    
    return scraped_articles

#--------------------------------
# Classification Functions
#--------------------------------

def classify_political_leaning(title, full_text, source_outlet=None, max_tokens=1000):
    """
    Classify the political leaning of an article using DeepSeek API.
    """
    # Truncate the text if it's too long
    if full_text and len(full_text) > max_tokens * 4:  # rough estimation of tokens
        text_sample = full_text[:max_tokens * 4]
        text_to_analyze = f"{title}\n\nExcerpt from article:\n{text_sample}..."
    else:
        text_to_analyze = f"{title}\n\n{full_text}"
    
    outlet_info = f"Publication: {source_outlet}\n\n" if source_outlet else ""
    
    prompt = f"""Analyze the following news article and determine its political leaning.
{outlet_info}
Article: {text_to_analyze}

Based ONLY on the content of this article, classify it into one of the following categories:
- Left
- Lean Left
- Center
- Lean Right
- Right

Consider factors such as:
- Language and framing
- Topics emphasized
- Sources quoted
- Overall narrative and tone

Provide ONLY the category name as your answer (Left, Lean Left, Center, Lean Right, or Right). 
Do not include any explanations or additional text.
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert political analyst who can accurately determine the political leaning of news articles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent results
            max_tokens=10,    # We only need a short answer
            stream=False
        )
        
        result = response.choices[0].message.content.strip()
        
        # Normalize the result to match our categories
        for category in CATEGORIES:
            if category.lower() in result.lower():
                return category
                
        # If no direct match, try to find closest match
        if "far left" in result.lower() or "extreme left" in result.lower():
            return "Left"
        elif "far right" in result.lower() or "extreme right" in result.lower():
            return "Right"
        
        # Default to Center if we can't determine
        return "Center"
        
    except Exception as e:
        print(f"Error classifying article: {str(e)}")
        return "Error"

def classify_article(article):
    """
    Classify a single article's political leaning.
    
    Args:
        article: A dictionary containing article data
    
    Returns:
        The updated article with classification added
    """
    # Skip if already classified with valid data
    if 'ai_political_leaning' in article and article['ai_political_leaning'] not in ["", "No content", "Error"]:
        return article
    
    # Get required fields
    title = article.get('title', '')
    full_text = article.get('full_text', '')
    source_outlet = article.get('source_outlet', '')
    
    # Skip articles with no text content
    if not full_text or full_text in ["Not available", "Failed to extract article text"]:
        article['ai_political_leaning'] = "No content"
        article['match_with_source'] = False
        return article
    
    # Classify the article
    classification = classify_political_leaning(title, full_text, source_outlet)
    
    # Store the result
    article['ai_political_leaning'] = classification
    
    # Compare with existing rating if available
    source_rating = article.get('source_rating', '')
    if source_rating:
        # Normalize source_rating to match our categories
        normalized_rating = source_rating
        if source_rating == "Far Left":
            normalized_rating = "Left"
        elif source_rating == "Far Right":
            normalized_rating = "Right"
            
        article['match_with_source'] = classification == normalized_rating
    else:
        article['match_with_source'] = None
    
    return article

#--------------------------------
# Data Management Functions
#--------------------------------

def load_existing_articles(file_path):
    """
    Load existing articles from a JSON file if it exists.
    Returns an empty list if the file doesn't exist.
    """
    # Debug information
    abs_path = os.path.abspath(file_path)
    print(f"DEBUG: Attempting to load articles from: {abs_path}")
    print(f"DEBUG: File exists: {os.path.exists(file_path)}")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"DEBUG: Successfully loaded {len(data)} articles")
                return data
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
    
    # Only add articles that don't already exist
    unique_new_articles = []
    for article in new_articles:
        if article['link'] not in existing_links:
            unique_new_articles.append(article)
            existing_links[article['link']] = article
    
    # Return the merged list of articles
    return existing_articles + unique_new_articles

def save_articles(articles, json_path, csv_path):
    """Save the articles to both JSON and CSV formats."""
    # Save as JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4)
    print(f"JSON file saved as {json_path}")
    
    # Save as CSV
    df = pd.DataFrame(articles)
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved as {csv_path}")

def print_classification_stats(articles):
    """Print statistics about the political classification distribution."""
    ratings = [a.get('ai_political_leaning', '') for a in articles 
              if a.get('ai_political_leaning', '') not in ["", "No content", "Error"]]
    
    if not ratings:
        print("No classified articles found.")
        return
    
    print("\nClassification Distribution:")
    for category in CATEGORIES:
        count = ratings.count(category)
        percentage = count / len(ratings) * 100 if ratings else 0
        print(f"{category}: {count} ({percentage:.1f}%)")
    
    # Match statistics
    matches = [a.get('match_with_source', False) for a in articles 
              if a.get('match_with_source', None) is not None]
    
    if matches:
        match_rate = matches.count(True) / len(matches) * 100
        print(f"\nMatch with source ratings: {matches.count(True)}/{len(matches)} ({match_rate:.1f}%)")

#--------------------------------
# Main Workflows
#--------------------------------

def initial_setup(url=DEFAULT_URL, json_path=DEFAULT_JSON_PATH, csv_path=DEFAULT_CSV_PATH):
    """
    Initial setup workflow:
    1. Scrape AllSides
    2. Extract full text for all articles
    3. Classify all articles
    4. Save the results
    """
    print(f"Starting initial setup from {url}")
    
    # Check if we already have existing data
    existing_articles = load_existing_articles(json_path)
    if existing_articles:
        print(f"Found {len(existing_articles)} existing articles in {json_path}")
        print("Initial setup will merge with and update existing data")
    
    # Scrape articles
    print(f"Scraping articles from {url}...")
    new_articles = scrape_allsides_page(url)
    print(f"Scraped {len(new_articles)} articles from AllSides")
    
    # Merge with existing (if any)
    all_articles = merge_articles(existing_articles, new_articles)
    print(f"Total articles after merging: {len(all_articles)}")
    
    # Classify all articles
    print("\nClassifying all articles for political leaning...")
    for article in tqdm(all_articles, desc="Classifying"):
        classify_article(article)
        time.sleep(random.uniform(0.5, 1.0))  # Delay to avoid rate limits
    
    # Save results
    save_articles(all_articles, json_path, csv_path)
    
    # Print statistics
    print_classification_stats(all_articles)
    
    print("\nInitial setup complete!")

def update_articles(url=DEFAULT_URL, json_path=DEFAULT_JSON_PATH, csv_path=DEFAULT_CSV_PATH):
    """
    Update workflow:
    1. Load existing articles
    2. Scrape new articles from AllSides
    3. Extract full text only for new articles
    4. Classify only new articles
    5. Save the updated dataset
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
        # Classify new articles
        print("Classifying new articles...")
        for article in tqdm(truly_new_articles, desc="Classifying new articles"):
            classify_article(article)
            time.sleep(random.uniform(0.5, 1.0))  # Delay to avoid rate limits
        
        # Merge and save
        all_articles = merge_articles(existing_articles, truly_new_articles)
        save_articles(all_articles, json_path, csv_path)
        
        # Print details of new articles
        print("\nNew articles added:")
        for article in truly_new_articles:
            print(f"Title: {article['title']}")
            print(f"Source: {article['source_outlet']} ({article['source_rating']})")
            if 'ai_political_leaning' in article:
                print(f"AI Classification: {article['ai_political_leaning']}")
            print(f"Full Text Preview: {article['full_text'][:100]}..." if article.get('full_text') else "No text extracted")
            print("-" * 40)
        
        # Print overall statistics 
        print_classification_stats(all_articles)
    else:
        print("No new articles found. Dataset is already up to date.")
    
    print("\nUpdate complete!")

#--------------------------------
# Main
#--------------------------------

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AllSides News Processor - Scrape, extract text, and classify political bias')
    
    # Main modes
    parser.add_argument('--initial-setup', action='store_true',
                      help='Run full initial setup (scrape, extract text, classify all)')
    parser.add_argument('--update', action='store_true',
                      help='Update mode - only process new articles (default if no mode specified)')
    
    # Common options
    parser.add_argument('--url', default=DEFAULT_URL,
                      help=f'URL to scrape (default: {DEFAULT_URL})')
    parser.add_argument('--json-path', default=DEFAULT_JSON_PATH,
                      help=f'Path to save/load JSON file (default: {DEFAULT_JSON_PATH})')
    parser.add_argument('--csv-path', default=DEFAULT_CSV_PATH,
                      help=f'Path to save CSV file (default: {DEFAULT_CSV_PATH})')
    
    args = parser.parse_args()
    
    # Default to update mode if no mode specified
    if not (args.initial_setup or args.update):
        args.update = True
    
    if args.initial_setup:
        initial_setup(args.url, args.json_path, args.csv_path)
    elif args.update:
        update_articles(args.url, args.json_path, args.csv_path) 