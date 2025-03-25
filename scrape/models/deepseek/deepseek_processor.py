#!/usr/bin/env python3
"""
DeepSeek Processor - Classify articles using DeepSeek API

This script:
1. Loads articles from the central database
2. Classifies their political leaning using DeepSeek API
3. Saves the classifications to model-specific files
"""

import json
import pandas as pd
import time
import random
import os
import sys
from tqdm import tqdm
from openai import OpenAI

# Try to load .env file if dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file if it exists
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. Using environment variables directly.")
    print("To use a .env file, install python-dotenv: pip install python-dotenv")

# DeepSeek API configuration - Read from environment variable
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("Warning: DEEPSEEK_API_KEY environment variable is not set.")
    print("Please set it using:")
    print("export DEEPSEEK_API_KEY='your-api-key-here'")
    print("or add it to your .env file and load it with a package like python-dotenv.")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Configure the OpenAI client to use DeepSeek
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# Define the categories we're interested in
CATEGORIES = ["Left", "Lean Left", "Center", "Lean Right", "Right"]

# File paths
# Use absolute paths for model files
DEEPSEEK_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_JSON_PATH = os.path.join(DEEPSEEK_DIR, "deepseek_articles.json")
DEFAULT_MODEL_CSV_PATH = os.path.join(DEEPSEEK_DIR, "deepseek_articles.csv")
DEFAULT_DATABASE_JSON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "articles_base.json")
DEFAULT_DATABASE_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "articles_base.csv")

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
    
    # Use a numerical rating scale instead of categorical labels
    prompt = f"""Instructions: Political Bias Scale from -6 to 6, where -6 to -3 is left, -3 to -1 is lean-left, -1 to 1 is center, 1 to 3 is lean-right, and 3 to 6 is right. Leaning categories (like lean-left or lean-right) should indicate content that leans towards that side but does not strongly align with it. 

A newspaper article is provided and you have to give it a decimal rating. Only analyze the article's content for language, framing, and overall tone to determine the political bias. Do NOT infer the news outlet or any external context beyond the article itself. If the bias is unclear, output the most appropriate rating based on the overall tone and content.

Article: {text_to_analyze}

Output only a number between -6 and 6 that represents the political rating. No other text.
"""

    try:
        # Make the API call to DeepSeek
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.1
        )
        
        # Extract the numerical rating from the response
        raw_response = response.choices[0].message.content.strip()
        
        # Try to extract a number from the response
        try:
            # Remove any non-numeric characters except for decimal point and minus sign
            cleaned_response = ''.join(c for c in raw_response if c.isdigit() or c == '.' or c == '-')
            rating = float(cleaned_response)
            
            # Map the numerical rating to the categorical labels
            if rating <= -3:
                return "Left"
            elif rating < -1:
                return "Lean Left"
            elif rating <= 1:
                return "Center"
            elif rating < 3:
                return "Lean Right"
            else:
                return "Right"
        except:
            # If we can't extract a valid number, check for category names in the response
            raw_response_lower = raw_response.lower()
            if "left" in raw_response_lower and "lean" in raw_response_lower:
                return "Lean Left"
            elif "right" in raw_response_lower and "lean" in raw_response_lower:
                return "Lean Right"
            elif "left" in raw_response_lower:
                return "Left"
            elif "right" in raw_response_lower:
                return "Right"
            elif "center" in raw_response_lower:
                return "Center"
            else:
                # Default to "No content" if we can't determine the classification
                return "No content"
        
    except Exception as e:
        print(f"Error classifying article: {e}")
        return "Error in classification"

def classify_article(article):
    """
    Classify an article and add the classification to the article object.
    """
    if 'ai_political_leaning' not in article or article['ai_political_leaning'] == "No content":
        title = article['title']
        full_text = article.get('full_text', "")
        source_outlet = article.get('source_outlet', "")
        
        # Skip articles without full text
        if not full_text or full_text == "Not available...":
            article['ai_political_leaning'] = "No content"
            article['match_with_source'] = False
            return article
        
        # Classify the article
        classification = classify_political_leaning(title, full_text, source_outlet)
        article['ai_political_leaning'] = classification
        
        # Check if the classification matches the source rating
        source_rating = article.get('source_rating', "")
        article['match_with_source'] = (classification == source_rating)
    
    return article

#--------------------------------
# Data Management Functions
#--------------------------------

def load_articles(file_path):
    """
    Load articles from a JSON file.
    """
    if not os.path.exists(file_path):
        print(f"Error: Database file {file_path} does not exist.")
        print("Please run update_database.py first to create the database.")
        sys.exit(1)
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading articles: {e}")
        sys.exit(1)

def load_existing_model_articles(file_path):
    """
    Load existing model-specific articles from a JSON file.
    Returns an empty list if the file doesn't exist.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading existing model articles: {e}")
            return []
    return []

def save_model_articles(articles, json_path, csv_path):
    """Save model-specific articles to JSON and CSV files."""
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

def print_classification_stats(articles):
    """Print statistics about political classifications and match rate."""
    # Count classifications
    classification_counts = {category: 0 for category in CATEGORIES}
    match_count = 0
    total_with_text = 0
    
    for article in articles:
        if 'full_text' in article and article['full_text'] != "Not available...":
            total_with_text += 1
            
        if 'ai_political_leaning' in article and article['ai_political_leaning'] in CATEGORIES:
            classification_counts[article['ai_political_leaning']] += 1
            
            if article.get('match_with_source', False):
                match_count += 1
    
    # Print distribution
    print("\nClassification Distribution:")
    total_classified = sum(classification_counts.values())
    for category, count in classification_counts.items():
        percentage = (count / total_classified * 100) if total_classified > 0 else 0
        print(f"{category}: {count} ({percentage:.1f}%)")
    
    # Print match rate
    if total_classified > 0:
        match_percentage = (match_count / total_classified * 100)
        print(f"\nMatch with source ratings: {match_count}/{total_classified} ({match_percentage:.1f}%)")
    
    print(f"\nFull text available for {total_with_text} out of {len(articles)} articles ({total_with_text/len(articles)*100:.1f}%)")

#--------------------------------
# Main Workflows
#--------------------------------

def classify_all(db_path=DEFAULT_DATABASE_JSON_PATH, model_json_path=DEFAULT_MODEL_JSON_PATH, model_csv_path=DEFAULT_MODEL_CSV_PATH):
    """
    Classify all articles in the database, preserving existing classifications.
    When --classify-all is specified, force reclassification of all articles.
    """
    # Check if the classify-all flag was passed
    force_classify_all = "--classify-all" in sys.argv
    
    # Load articles from database
    articles = load_articles(db_path)
    print(f"Loaded {len(articles)} articles from database")
    
    # Load existing model classifications
    existing_model_articles = load_existing_model_articles(model_json_path)
    print(f"Found {len(existing_model_articles)} existing classifications")
    
    # Create a dictionary of existing classifications for faster lookup
    existing_classifications = {article['link']: article for article in existing_model_articles}
    
    # Track which articles need classification
    articles_to_classify = []
    articles_to_preserve = []
    
    if force_classify_all:
        print("\nForce classifying ALL articles regardless of existing classifications")
        articles_to_classify = articles
    else:
        for article in articles:
            if article['link'] in existing_classifications:
                # Preserve the existing classification
                existing_article = existing_classifications[article['link']]
                article['ai_political_leaning'] = existing_article['ai_political_leaning']
                article['match_with_source'] = existing_article.get('match_with_source', False)
                articles_to_preserve.append(article)
            else:
                articles_to_classify.append(article)
    
        print(f"\nFound {len(articles_to_classify)} new articles to classify")
        print(f"Preserving {len(articles_to_preserve)} existing classifications")
    
    if articles_to_classify:
        # Classify new articles
        print("\nClassifying articles...")
        for article in tqdm(articles_to_classify, desc="Classifying articles"):
            article = classify_article(article)
            # Add a small delay to avoid rate limiting
            time.sleep(random.uniform(1.0, 2.0))
        
        # Combine preserved and new classifications if not force_classify_all
        if force_classify_all:
            all_articles = articles_to_classify
        else:
            all_articles = articles_to_preserve + articles_to_classify
    else:
        print("No new articles to classify")
        all_articles = articles_to_preserve
    
    # Save the combined results
    save_model_articles(all_articles, model_json_path, model_csv_path)
    
    # Print statistics
    print_classification_stats(all_articles)
    
    return all_articles

def classify_new(db_path=DEFAULT_DATABASE_JSON_PATH, model_json_path=DEFAULT_MODEL_JSON_PATH, model_csv_path=DEFAULT_MODEL_CSV_PATH):
    """
    Classify only new articles that haven't been classified yet.
    """
    return classify_all(db_path, model_json_path, model_csv_path)

#--------------------------------
# Main
#--------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSeek Political Leaning Classifier")
    
    # File paths
    parser.add_argument('--db-path', default=DEFAULT_DATABASE_JSON_PATH,
                      help=f'Path to database JSON file (default: {DEFAULT_DATABASE_JSON_PATH})')
    parser.add_argument('--model-json-path', default=DEFAULT_MODEL_JSON_PATH,
                      help=f'Path to save model JSON file (default: {DEFAULT_MODEL_JSON_PATH})')
    parser.add_argument('--model-csv-path', default=DEFAULT_MODEL_CSV_PATH,
                      help=f'Path to save model CSV file (default: {DEFAULT_MODEL_CSV_PATH})')
    
    # Classification mode
    parser.add_argument('--classify-all', action='store_true',
                      help='Classify all articles (including already classified ones)')
    
    args = parser.parse_args()
    
    try:
        # Note: The flag detection is now handled in the classify_all function itself
        classify_all(args.db_path, args.model_json_path, args.model_csv_path)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}") 