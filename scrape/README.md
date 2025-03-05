# AllSides News Scraper and Political Bias Classifier

This project scrapes news articles from AllSides.com, extracts their full text, and classifies their political leaning using AI. It maintains a dataset of articles that grows over time as you collect more data.

## Requirements

- Python 3.7+
- Required packages: pandas, tqdm, openai, BeautifulSoup, newspaper3k, requests, cloudscraper

Install the required packages using pip:
```bash
pip install pandas tqdm openai beautifulsoup4 newspaper3k requests cloudscraper
```

## Usage

The project provides a single script `news_processor.py` with two main modes:

### 1. Initial Setup Mode

For your first run or when you want to start fresh:

```bash
python news_processor.py --initial-setup
```

This will:
- Scrape all articles from AllSides
- Extract their full text
- Classify their political leaning using the DeepSeek API
- Save everything to both JSON and CSV

### 2. Update Mode (Default)

For regular updates - only processes new articles:

```bash
python news_processor.py
# or
python news_processor.py --update
```

This will:
- Load your existing articles
- Find only new articles from AllSides
- Extract full text for only those new articles
- Classify only the new articles
- Merge with your existing dataset
- Never re-process articles you already have

## Command-Line Arguments

`news_processor.py` accepts the following arguments:

- `--initial-setup`: Run full initial setup (scrape, extract text, classify all)
- `--update`: Update mode - only process new articles (default if no mode specified)
- `--url`: URL to scrape (default: "https://www.allsides.com/unbiased-balanced-news")
- `--json-path`: Path to save/load JSON file (default: "allsides_articles.json")
- `--csv-path`: Path to save CSV file (default: "allsides_articles.csv")

## Understanding the Classification Process

The political leaning classifier uses the DeepSeek API to analyze article content and determine its bias across five categories:
- Left
- Lean Left
- Center
- Lean Right
- Right

The system also compares the AI classification with AllSides' own source rating to see how often they match.

## How It Works

1. **Article Scraping**: Articles are scraped from AllSides, including metadata like title, source outlet, and source rating.
2. **Full Text Extraction**: The system extracts the full text of each article using the original source URL.
3. **Deduplication**: The system checks for duplicates to prevent re-adding articles you've already scraped.
4. **Classification**: The system uses the DeepSeek API to classify the political leaning of articles.
5. **Dataset Management**: All articles are saved in both JSON and CSV formats for easy analysis.

## Troubleshooting

- **API Rate Limits**: The DeepSeek API has rate limits. The script includes random delays to help avoid them.
- **Text Extraction Failures**: Some articles may be behind paywalls or have anti-scraping measures. The system will mark these as failed.
- **Import Errors**: Make sure all required packages are installed.

## Data Files

- **allsides_articles.json**: Main JSON file containing all article data.
- **allsides_articles.csv**: CSV version of the same data for easy import into analysis tools. 