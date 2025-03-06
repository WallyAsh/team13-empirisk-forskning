# AllSides News Scraper and Multi-Model Political Bias Classifier

This project scrapes news articles from AllSides.com, extracts their full text, and classifies their political leaning using multiple AI models. Each model has its own independent classification system.

## Features

- Central database of articles with full text
- Multiple independent classifier models:
  - DeepSeek API (implemented)
  - ChatGPT (future)
  - Gemini (future)
  - Add your own models

## Project Organization

```
scrape/
├── data/                       # Central database
│   ├── articles_base.json
│   └── articles_base.csv
├── models/
│   ├── deepseek/               # DeepSeek classifier
│   │   ├── deepseek_processor.py
│   │   ├── deepseek_articles.json
│   │   └── deepseek_articles.csv
│   ├── chatgpt/                # Future classifier
│   └── gemini/                 # Future classifier
├── update_database.py          # Database manager
└── classify.py                 # Main launcher
```

## Quick Start

```bash
# Navigate to scrape directory
cd scrape

# First time setup
python classify.py --initial-setup

# Update database and classify with DeepSeek
python classify.py --update-db --model deepseek

# Just classify with DeepSeek (no database update)
python classify.py --model deepseek
```

For detailed instructions on setup and usage, see [scrape/README.md](scrape/README.md). 