#!/usr/bin/env python3
"""
Script to reclassify the selected subset of articles using DeepSeek.

This script:
1. Loads the final subset of high-quality articles
2. Reclassifies them using the DeepSeek API with a numerical rating scale
3. Saves both the numerical rating and categorical classification
4. Analyzes the classification results
"""

import json
import pandas as pd
import os
import time
import random
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

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("Error: DEEPSEEK_API_KEY environment variable is not set.")
    sys.exit(1)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Configure the OpenAI client to use DeepSeek
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# File paths
DEFAULT_INPUT_PATH = 'analysis_subset/final_subset.json'
DEFAULT_OUTPUT_PATH = 'analysis_subset/classified_subset.json'
DEFAULT_RESULTS_PATH = 'analysis_subset/classification_results.csv'

def load_articles(file_path):
    """Load articles from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading articles: {e}")
        return []

def save_articles(articles, file_path):
    """Save articles to JSON file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(articles)} articles to {file_path}")

def classify_article(article):
    """
    Classify an article using the DeepSeek API with a numerical rating scale.
    """
    title = article.get('title', '')
    full_text = article.get('full_text', '')
    
    if not full_text or full_text == "Not available..." or full_text == "Not available":
        return {"ai_political_rating": None, "ai_political_leaning": "No content"}
    
    # Truncate the text if it's too long
    max_tokens = 1000
    if len(full_text) > max_tokens * 4:  # rough estimation of tokens
        text_sample = full_text[:max_tokens * 4]
        text_to_analyze = f"{title}\n\nExcerpt from article:\n{text_sample}..."
    else:
        text_to_analyze = f"{title}\n\n{full_text}"
    
    # Numerical rating scale prompt
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
        
        try:
            # Clean the response to get a numerical value
            cleaned_response = ''.join(c for c in raw_response if c.isdigit() or c == '.' or c == '-')
            rating = float(cleaned_response)
            
            # Map the numerical rating to the categorical labels
            if rating <= -3:
                category = "Left"
            elif rating < -1:
                category = "Lean Left"
            elif rating <= 1:
                category = "Center"
            elif rating < 3:
                category = "Lean Right"
            else:
                category = "Right"
            
            return {"ai_political_rating": rating, "ai_political_leaning": category}
            
        except ValueError:
            print(f"Could not parse rating from response: {raw_response}")
            return {"ai_political_rating": None, "ai_political_leaning": "Error"}
            
    except Exception as e:
        print(f"Error classifying article: {e}")
        return {"ai_political_rating": None, "ai_political_leaning": "Error"}

def analyze_results(articles):
    """
    Analyze the classification results.
    """
    # Create a DataFrame for analysis
    data = []
    for article in articles:
        data.append({
            'title': article.get('title', ''),
            'source_outlet': article.get('source_outlet', ''),
            'source_rating': article.get('source_rating', ''),
            'source_rating_value': article.get('source_rating_value', ''),
            'ai_political_rating': article.get('ai_political_rating', ''),
            'ai_political_leaning': article.get('ai_political_leaning', ''),
            'match_with_source': article.get('ai_political_leaning', '') == article.get('source_rating', '')
        })
    
    df = pd.DataFrame(data)
    
    # Save results to CSV
    df.to_csv(DEFAULT_RESULTS_PATH, index=False)
    print(f"Saved analysis results to {DEFAULT_RESULTS_PATH}")
    
    # Calculate overall match rate
    total_articles = len(df)
    matches = df['match_with_source'].sum()
    match_rate = (matches / total_articles) * 100 if total_articles > 0 else 0
    
    # Calculate statistics by source rating
    source_stats = df.groupby('source_rating').agg({
        'match_with_source': ['count', 'sum', lambda x: (x.sum() / len(x)) * 100 if len(x) > 0 else 0]
    })
    
    # Print analysis results
    print("\nClassification Analysis:")
    print("-" * 40)
    print(f"Total articles: {total_articles}")
    print(f"Exact matches: {matches} ({match_rate:.1f}%)")
    
    print("\nAI Political Leaning Distribution:")
    leaning_counts = df['ai_political_leaning'].value_counts()
    for leaning, count in leaning_counts.items():
        percentage = (count / total_articles) * 100
        print(f"  {leaning}: {count} ({percentage:.1f}%)")
    
    print("\nMatch Rate by Source Rating:")
    for source_rating, (count, matches, match_rate) in source_stats.iterrows():
        print(f"  {source_rating}: {matches}/{count} ({match_rate:.1f}%)")
    
    return df

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reclassify the selected subset of articles using DeepSeek")
    parser.add_argument('--input-path', default=DEFAULT_INPUT_PATH,
                      help=f'Path to the input JSON file (default: {DEFAULT_INPUT_PATH})')
    parser.add_argument('--output-path', default=DEFAULT_OUTPUT_PATH,
                      help=f'Path to the output JSON file (default: {DEFAULT_OUTPUT_PATH})')
    args = parser.parse_args()
    
    print(f"Loading articles from {args.input_path}...")
    articles = load_articles(args.input_path)
    
    if not articles:
        print("No articles found. Exiting.")
        return
    
    print(f"Loaded {len(articles)} articles")
    
    # Classify each article
    print("\nClassifying articles with DeepSeek...")
    for i, article in enumerate(tqdm(articles, desc="Classifying")):
        classification = classify_article(article)
        
        # Update the article with the new classification
        article.update(classification)
        
        # Add a delay to avoid rate limiting
        time.sleep(random.uniform(1.0, 2.0))
    
    # Save the classified articles
    save_articles(articles, args.output_path)
    
    # Analyze the results
    analyze_results(articles)

if __name__ == "__main__":
    main() 