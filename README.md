# AllSides News Scraper and Political Bias Classifier

This project scrapes news articles from AllSides.com, extracts their full text, and classifies their political leaning using AI.

## Features

- Scrapes articles from AllSides.com
- Extracts full text of articles from their original sources
- Classifies political bias using DeepSeek API
- Maintains a growing dataset of articles with classifications
- Intelligently processes only new articles during updates

## Quick Start

```bash
# Navigate to the scrape directory
cd scrape

# For initial setup:
python news_processor.py --initial-setup

# For regular updates (only processes new articles):
python news_processor.py
```

For detailed instructions, available options, and explanations of how the system works, please see the [README.md in the scrape directory](scrape/README.md). 