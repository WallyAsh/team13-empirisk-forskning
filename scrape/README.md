# AllSides News Scraper and Multi-Model Bias Classifier

This project scrapes articles from AllSides.com, extracts their full text, and classifies them according to their political bias using multiple AI models.

## Project Organization

The project is organized with a central database and model-specific classifiers:

```
scrape/
├── data/
│   ├── articles_base.json       # Central database without classifications
│   └── articles_base.csv        # CSV version of base data
├── models/
│   ├── deepseek/
│   │   ├── deepseek_processor.py   # DeepSeek-specific classifier
│   │   ├── deepseek_articles.json  # DeepSeek classifications
│   │   └── deepseek_articles.csv   # CSV version with same data
│   ├── chatgpt/                 # Future model implementation
│   └── gemini/                  # Future model implementation
├── update_database.py           # Central database manager
└── classify.py                  # Main launcher script
```

## Setup

1. Make sure you have all required packages:
   ```
   pip install requests bs4 newspaper3k cloudscraper tqdm pandas openai
   ```

2. Update your API keys in the model processors:
   - Edit `models/deepseek/deepseek_processor.py` to update DeepSeek API key

## API Key Setup

### DeepSeek API

For classification with the DeepSeek model, you'll need to set up your API key:

1. Copy the example environment file:
   ```
   cp models/deepseek/.env.example models/deepseek/.env
   ```

2. Edit the `.env` file and replace `your_api_key_here` with your actual DeepSeek API key.

3. Install python-dotenv (optional but recommended):
   ```
   pip install python-dotenv
   ```

Alternatively, you can set the API key as an environment variable:
```
export DEEPSEEK_API_KEY="your-api-key-here"
```

## Usage

### Using the Main Launcher

For the easiest workflow, use the main `classify.py` launcher:

```bash
# First time setup - scrape articles and set up the database  (! DO NOT RUN RIGHT NOW, BECAUSE WE ALREADY HAVE CREATED A DATABASE)
python classify.py --initial-setup

# Regular database updates
python classify.py --update-db

# Classify NEW articles with DeepSeek
python classify.py --model deepseek

# List available models
python classify.py --list-models

# Update database and classify in one command
python classify.py --update-db --model deepseek

# Classify all articles (not just new ones)
python classify.py --model deepseek --classify-all

# Extract missing text from articles
python update_database.py --enhanced-extract
```

### Advanced: Direct Script Usage

You can also use the scripts directly:

```bash
# Update the central database
python update_database.py

# Classify with DeepSeek
cd models/deepseek
python deepseek_processor.py
```

## How It Works

1. **Database Management** (`update_database.py`):
   - Scrapes articles from AllSides
   - Extracts full text from original sources
   - Maintains a central database in `data/articles_base.json`
   - No classification (models will do that)

2. **Article Classification** (model processors):
   - Each model (DeepSeek, ChatGPT, etc.) has its own processor
   - Processors read from the central database
   - They classify articles based on content
   - Results are saved in model-specific files
   - No scraping or text extraction (database handles that)

3. **Multi-Model Launcher** (`classify.py`):
   - Provides a unified interface for all operations
   - Can update the database and run classifiers in one command
   - Makes it easy to use multiple models

## Adding New Models

To add a new model (e.g., ChatGPT):

1. Create a new folder: `models/chatgpt/`
2. Copy and adapt `deepseek_processor.py` to create `chatgpt_processor.py`
3. Update the API configuration and classification function
4. Add the model to the `AVAILABLE_MODELS` list in `classify.py`

## Data Files

- **Central Database**:
  - `data/articles_base.json` - All article data without classifications
  - `data/articles_base.csv` - CSV version of the same data

- **Model-Specific Data** (e.g., DeepSeek):
  - `models/deepseek/deepseek_articles.json` - DeepSeek classifications
  - `models/deepseek/deepseek_articles.csv` - CSV version

## Future Development

The `classify.py` script in the root directory will be developed in the future to act as a unified launcher for all model-specific processors. 