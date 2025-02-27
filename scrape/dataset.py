import cloudscraper
from bs4 import BeautifulSoup
import time
import pandas as pd

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
        # Get the first anchor (article link) in the news-item block
        link_tag = item.find('a')
        if link_tag is None:
            continue

        link = link_tag.get('href', 'No link found')
        if link.startswith('/'):
            link = "https://www.allsides.com" + link

        # Extract title from the nested div with class "news-title"
        title_div = link_tag.find('div', class_='news-title')
        title = title_div.get_text(strip=True) if title_div else link_tag.get_text(strip=True)
        
        # Extract the source information from the anchor with class 'source-area'
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

        # Get the original source by scraping the article's detail page.
        original_source = get_original_source(link, scraper, headers)
        # Optional delay to avoid overwhelming the server with requests.
        time.sleep(0.5)
        
        scraped_articles.append({
            "title": title,
            "link": link,
            "source_outlet": outlet,
            "source_url": source_url,
            "source_rating": rating,
            "original_source": original_source
        })
    return scraped_articles

if __name__ == '__main__':
    url = "https://www.allsides.com/unbiased-balanced-news"
    articles = scrape_allsides_page(url)
    
    # Print each article's details
    for article in articles:
        print(f"Title: {article['title']}")
        print(f"Link: {article['link']}")
        print(f"Source Outlet: {article['source_outlet']}")
        print(f"Source URL: {article['source_url']}")
        print(f"Source Rating: {article['source_rating']}")
        print(f"Original Source: {article['original_source']}")
        print("-" * 40)
    
    df = pd.DataFrame(articles)
    
    # Save as CSV
    csv_filename = "allsides_articles.csv"
    df.to_csv(csv_filename, index=False)
    print(f"CSV file saved as {csv_filename}")
    
    # Save as JSON
    json_filename = "allsides_articles.json"
    df.to_json(json_filename, orient="records", indent=4)
    print(f"JSON file saved as {json_filename}")
