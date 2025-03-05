import requests
import time
import random
import json
import pandas as pd
import os
from newspaper import Article
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import concurrent.futures
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("article_retrieval.log"),
        logging.StreamHandler()
    ]
)

# Common headers to mimic a browser
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:95.0) Gecko/20100101 Firefox/95.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def extract_with_newspaper(url, timeout=10):
    """
    Extract article text using newspaper3k library
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        if not article.text or len(article.text.strip()) < 100:
            return None
            
        return article.text
    except Exception as e:
        logging.error(f"Newspaper extraction failed for {url}: {str(e)}")
        return None

def extract_with_bs4(url, timeout=10):
    """
    Extract article text using BeautifulSoup
    """
    try:
        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Referer": "https://www.google.com/",
            "DNT": "1",
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Common article content containers
        article_containers = soup.select("article, .article, .post, .content, .article-content, .post-content, .story-content, .entry-content, main")
        
        if article_containers:
            # Use the first found container
            content = article_containers[0].get_text(separator='\n', strip=True)
        else:
            # Fallback to body text if no container found
            paragraphs = soup.find_all('p')
            content = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40])
            
        # Basic cleaning
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'\s+', ' ', content)
        
        if not content or len(content.strip()) < 100:
            return None
            
        return content
    except Exception as e:
        logging.error(f"BS4 extraction failed for {url}: {str(e)}")
        return None

def extract_text_from_archive(url):
    """
    Try to extract article from Wayback Machine archive
    """
    try:
        domain = urlparse(url).netloc
        wayback_url = f"https://web.archive.org/web/2023/{url}"
        
        headers = {"User-Agent": get_random_user_agent()}
        response = requests.get(wayback_url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return None
            
        # Try to extract from archived page
        try:
            article = Article(wayback_url)
            article.download()
            article.parse()
            if article.text and len(article.text) > 100:
                return article.text
        except:
            pass
            
        # Fallback to BS4 on archived page
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Extract paragraphs from the main content
        paragraphs = soup.find_all('p')
        content = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 40])
        
        if not content or len(content) < 100:
            return None
            
        return content
    except Exception as e:
        logging.error(f"Archive extraction failed for {url}: {str(e)}")
        return None

def extract_article_with_fallbacks(url):
    """
    Try multiple methods to extract article text with fallbacks
    """
    logging.info(f"Attempting to extract text from: {url}")
    
    # Try newspaper3k first (most reliable when it works)
    content = extract_with_newspaper(url)
    if content:
        logging.info(f"✓ Successfully extracted with newspaper3k: {url}")
        return content
        
    # Try BeautifulSoup as fallback
    content = extract_with_bs4(url)
    if content:
        logging.info(f"✓ Successfully extracted with BeautifulSoup: {url}")
        return content
        
    # Try archive.org as last resort
    content = extract_text_from_archive(url)
    if content:
        logging.info(f"✓ Successfully extracted from web archive: {url}")
        return content
        
    logging.warning(f"× Failed to extract content from: {url}")
    return None

def process_articles_from_json(json_file, output_file=None, max_workers=5):
    """
    Process articles from a JSON file and attempt to retrieve missing text
    """
    # Load articles from JSON
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logging.info(f"Loaded {len(articles)} articles from {json_file}")
    except Exception as e:
        logging.error(f"Error loading JSON file: {e}")
        return False
    
    # Find articles with missing or failed text
    missing_articles = []
    for i, article in enumerate(articles):
        if ('full_text' not in article or 
            not article['full_text'] or 
            article['full_text'] in ["", "Not available", "Failed to extract article text", "Insufficient text extracted"] or
            article['full_text'].startswith("Failed to extract article text:")):
            if 'original_source' in article and article['original_source'] and article['original_source'] != "Not found":
                missing_articles.append((i, article))
    
    logging.info(f"Found {len(missing_articles)} articles with missing text")
    
    # Process articles with missing text using multiple threads
    updated_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_article = {
            executor.submit(extract_article_with_fallbacks, article['original_source']): (i, article) 
            for i, article in missing_articles
        }
        
        for future in concurrent.futures.as_completed(future_to_article):
            idx, article = future_to_article[future]
            url = article['original_source']
            
            try:
                text = future.result()
                if text:
                    articles[idx]['full_text'] = text
                    updated_count += 1
                    logging.info(f"Updated article {idx+1}/{len(missing_articles)}: {article['title']}")
                else:
                    logging.warning(f"Could not retrieve text for article {idx+1}/{len(missing_articles)}: {article['title']}")
            except Exception as e:
                logging.error(f"Error processing {url}: {str(e)}")
            
            # Sleep briefly to be nice to servers
            time.sleep(random.uniform(1.0, 2.0))
    
    logging.info(f"Successfully updated {updated_count} out of {len(missing_articles)} articles")
    
    # Save the updated articles
    if updated_count > 0:
        if not output_file:
            output_file = json_file
            
        # Create backup of original file
        backup_file = f"{json_file}.bak"
        try:
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=4)
            logging.info(f"Created backup at {backup_file}")
        except Exception as e:
            logging.error(f"Error creating backup: {e}")
            
        # Save updated file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=4)
            logging.info(f"Saved updated articles to {output_file}")
            
            # Also save as CSV
            csv_file = output_file.replace('.json', '.csv')
            pd.DataFrame(articles).to_csv(csv_file, index=False)
            logging.info(f"Saved updated articles to CSV: {csv_file}")
            
            return True
        except Exception as e:
            logging.error(f"Error saving updated file: {e}")
            return False
    else:
        logging.info("No articles were updated, files remain unchanged")
        return False

if __name__ == "__main__":
    json_file = "allsides_articles.json"
    process_articles_from_json(json_file, max_workers=3) 