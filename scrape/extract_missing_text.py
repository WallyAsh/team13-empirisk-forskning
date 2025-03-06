#!/usr/bin/env python3
#IN CASE IF RUNNING THE NEWS PROCESSOR FAILS TO EXTRACT THE TEXT, RUN THIS SCRIPT TO EXTRACT THE TEXT
import json
import time
import random
import requests
import cloudscraper
from newspaper import Article
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
import re

# File path
JSON_FILE_PATH = "allsides_articles.json"

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
    print(f"\nExtracting text from: {url}")
    
    # Method 1: Using newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        
        if text and len(text.strip()) > 200:  # Ensure we have meaningful content
            print("✓ Success using newspaper3k")
            return text
    except Exception as e:
        print(f"✗ newspaper3k extraction failed: {e}")
    
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
                print("✓ Success using requests+BeautifulSoup")
                return article_text
    except Exception as e:
        print(f"✗ BeautifulSoup extraction failed: {e}")
    
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
                print("✓ Success using cloudscraper")
                return article_text
    except Exception as e:
        print(f"✗ Cloudscraper extraction failed: {e}")
    
    print("✗ All extraction methods failed")
    return None

def process_articles_without_text():
    """Load articles, extract missing text, and save back to JSON."""
    # Load existing articles
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        print(f"Loaded {len(articles)} articles from {JSON_FILE_PATH}")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Find articles without text
    articles_without_text = [article for article in articles 
                            if 'full_text' not in article or not article['full_text'] 
                            or article['full_text'] == "Not available..."]
    
    print(f"Found {len(articles_without_text)} articles without text")
    
    # Process articles without text
    updated_count = 0
    for article in tqdm(articles_without_text, desc="Extracting text"):
        # Get the original source URL
        original_url = article.get('original_source')
        
        if not original_url or original_url == "Not found":
            print(f"No original source URL for article: {article['title']}")
            continue
        
        # Extract text from the original URL
        full_text = extract_article_text(original_url)
        
        if full_text:
            # Find the article in the main list and update it
            for main_article in articles:
                if main_article['link'] == article['link']:
                    main_article['full_text'] = full_text
                    updated_count += 1
                    break
        
        # Add a delay to avoid rate limiting
        time.sleep(random.uniform(2.0, 4.0))
    
    # Save the updated articles back to the JSON file
    with open(JSON_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
    
    print(f"\nUpdated {updated_count} out of {len(articles_without_text)} articles with missing text")
    print(f"Saved updates to {JSON_FILE_PATH}")

if __name__ == "__main__":
    process_articles_without_text() 