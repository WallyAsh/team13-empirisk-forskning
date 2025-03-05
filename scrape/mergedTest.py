import cloudscraper
from bs4 import BeautifulSoup
import time
import pandas as pd
import os
import json
from newspaper import Article

def extract_article_text(url):
    """
    Extract the text content from an article URL using newspaper3k library.
    Includes error handling for common issues.
    """
    try:
        article = Article(url)
        article.download()
        # Add a timeout to prevent hanging on problematic downloads
        article.parse()
        
        # If text is empty, try an alternative approach
        if not article.text or len(article.text.strip()) < 20:  # Less than 20 chars is likely not a valid article
            return "Insufficient text extracted"
            
        return article.text
    except Exception as e:
        print(f"Error extracting text from {url}: {str(e)}")
        return f"Failed to extract article text: {str(e)[:100]}"

def get_original_source(article_url, scraper, headers):
    """
    Fetch the article detail page and extract the 'Read full story'
    link which is assumed to be the original source.
    """
    try:
        detail_response = scraper.get(article_url, headers=headers, timeout=10)
        if detail_response.status_code != 200:
            return "Not found"
        detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
        read_more = detail_soup.find('div', class_='read-more-story')
        if read_more:
            original_anchor = read_more.find('a', href=True)
            if original_anchor:
                return original_anchor['href']
        return "Not found"
    except Exception as e:
        return "Not found"

def scrape_allsides_page(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
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
            if rating == "Not rated":
                rating = "Far Right"
        else:
            outlet = "No source found"
            source_url = ""
            rating = "Not rated"

        original_source = get_original_source(link, scraper, headers)
        # Extract full text from the original source if available
        if original_source != "Not found":
            full_text = extract_article_text(original_source)
        else:
            full_text = "Not available"

        time.sleep(0.5)
        
        scraped_articles.append({
            "title": title,
            "link": link,
            "source_outlet": outlet,
            "source_url": source_url,
            "source_rating": rating,
            "original_source": original_source,
            "full_text": full_text
        })
    return scraped_articles

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
    
    # Only add articles that don't already exist
    unique_new_articles = []
    for article in new_articles:
        if article['link'] not in existing_links:
            # Ensure all articles have the full_text field
            if 'full_text' not in article and 'original_source' in article:
                # Try to extract article text if it's missing but we have the original source
                if article['original_source'] != "Not found":
                    article['full_text'] = extract_article_text(article['original_source'])
                else:
                    article['full_text'] = "Not available"
            
            unique_new_articles.append(article)
        else:
            # If the article already exists but doesn't have full text, 
            # and the new one does, update the existing article
            existing_article = existing_links[article['link']]
            if ('full_text' not in existing_article or 
                existing_article['full_text'] == "Not available" or 
                existing_article['full_text'] == "Failed to extract article text"):
                if 'full_text' in article and article['full_text'] not in ["Not available", "Failed to extract article text"]:
                    existing_article['full_text'] = article['full_text']
    
    # Combine existing and new unique articles
    combined = list(existing_links.values()) + unique_new_articles
    
    return combined

if __name__ == '__main__':
    url = "https://www.allsides.com/unbiased-balanced-news"
    new_articles = scrape_allsides_page(url)
    
    # Define file paths
    json_path = "scrape/allsides_articles.json"
    csv_path = "scrape/allsides_articles.csv"
    
    # Load existing articles
    existing_articles = load_existing_articles(json_path)
    print(f"Loaded {len(existing_articles)} existing articles")
    
    # Merge with new articles
    all_articles = merge_articles(existing_articles, new_articles)
    print(f"Added {len(new_articles)} new unique articles")
    print(f"Total articles in dataset: {len(all_articles)}")
    
    # Print details of new articles
    print("\nNew articles added:")
    for article in new_articles:
        print(f"Title: {article['title']}")
        print(f"Link: {article['link']}")
        print(f"Source Outlet: {article['source_outlet']}")
        print(f"Source Rating: {article['source_rating']}")
        print("Full Text Preview:", article['full_text'][:100] if 'full_text' in article else "No text extracted", "...")
        print("-" * 40)
    
    # Save the data as CSV and JSON
    df = pd.DataFrame(all_articles)
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved as {csv_path}")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, indent=4)
    print(f"JSON file saved as {json_path}")
