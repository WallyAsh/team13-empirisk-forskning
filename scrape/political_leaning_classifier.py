import json
import pandas as pd
import time
import random
from openai import OpenAI
import os
from tqdm import tqdm
import numpy as np

# DeepSeek API configuration
DEEPSEEK_API_KEY = "sk-739ab05b7a6d4853bd6615e48387cd2f"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Configure the OpenAI client to use DeepSeek
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# Define the categories we're interested in
CATEGORIES = ["Left", "Lean Left", "Center", "Lean Right", "Right"]

def classify_political_leaning(title, full_text, source_outlet=None, max_tokens=1000):
    """
    Classify the political leaning of an article using DeepSeek API.
    
    Args:
        title: The article title
        full_text: The full text of the article
        source_outlet: The name of the publication (optional)
        max_tokens: Maximum text length to send to the API
    
    Returns:
        A string indicating the political leaning: Left, Lean Left, Center, Lean Right, or Right
    """
    # Truncate the text if it's too long
    if full_text and len(full_text) > max_tokens * 4:  # rough estimation of tokens
        text_sample = full_text[:max_tokens * 4]
        text_to_analyze = f"{title}\n\nExcerpt from article:\n{text_sample}..."
    else:
        text_to_analyze = f"{title}\n\n{full_text}"
    
    outlet_info = f"Publication: {source_outlet}\n\n" if source_outlet else ""
    
    prompt = f"""Analyze the following news article and determine its political leaning.
{outlet_info}
Article: {text_to_analyze}

Based ONLY on the content of this article, classify it into one of the following categories:
- Left
- Lean Left
- Center
- Lean Right
- Right

Consider factors such as:
- Language and framing
- Topics emphasized
- Sources quoted
- Overall narrative and tone

Provide ONLY the category name as your answer (Left, Lean Left, Center, Lean Right, or Right). 
Do not include any explanations or additional text.
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert political analyst who can accurately determine the political leaning of news articles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent results
            max_tokens=10,    # We only need a short answer
            stream=False
        )
        
        result = response.choices[0].message.content.strip()
        
        # Normalize the result to match our categories
        for category in CATEGORIES:
            if category.lower() in result.lower():
                return category
                
        # If no direct match, try to find closest match
        if "far left" in result.lower() or "extreme left" in result.lower():
            return "Left"
        elif "far right" in result.lower() or "extreme right" in result.lower():
            return "Right"
        
        # Default to Center if we can't determine
        return "Center"
        
    except Exception as e:
        print(f"Error classifying article: {str(e)}")
        return "Error"


def process_articles(json_file, output_file=None, limit=None):
    """
    Process articles from a JSON file and classify their political leaning.
    
    Args:
        json_file: Path to the JSON file containing articles
        output_file: Path to save the results (default: adds '_classified' to input filename)
        limit: Maximum number of articles to process (default: None for all)
    """
    # Load articles from JSON
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        print(f"Loaded {len(articles)} articles from {json_file}")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return False
    
    # Only process a limited number if specified
    if limit and limit < len(articles):
        print(f"Processing only {limit} articles out of {len(articles)}")
        # Use random subset for more diverse analysis
        articles_to_process = random.sample(articles, limit)
    else:
        articles_to_process = articles
    
    # Process each article
    success_count = 0
    for i, article in enumerate(tqdm(articles_to_process, desc="Classifying articles")):
        if i > 0:
            # Random delay between requests to avoid rate limits
            time.sleep(random.uniform(0.5, 2.0))
        
        # Get required fields
        title = article.get('title', '')
        full_text = article.get('full_text', '')
        source_outlet = article.get('source_outlet', '')
        
        # Skip articles with no text content
        if not full_text or full_text in ["Not available", "Failed to extract article text"]:
            article['ai_political_leaning'] = "No content"
            article['match_with_source'] = False
            continue
        
        # Classify the article
        classification = classify_political_leaning(title, full_text, source_outlet)
        
        # Store the result
        article['ai_political_leaning'] = classification
        
        # Compare with existing rating if available
        source_rating = article.get('source_rating', '')
        if source_rating:
            # Normalize source_rating to match our categories
            normalized_rating = source_rating
            if source_rating == "Far Left":
                normalized_rating = "Left"
            elif source_rating == "Far Right":
                normalized_rating = "Right"
                
            article['match_with_source'] = classification == normalized_rating
        else:
            article['match_with_source'] = None
        
        success_count += 1
    
    print(f"Successfully classified {success_count} out of {len(articles_to_process)} articles")
    
    # Calculate statistics
    ratings = [a.get('ai_political_leaning', '') for a in articles_to_process if a.get('ai_political_leaning', '') != "No content"]
    matches = [a.get('match_with_source', False) for a in articles_to_process if a.get('match_with_source', None) is not None]
    
    print("\nClassification Distribution:")
    for category in CATEGORIES:
        count = ratings.count(category)
        percentage = count / len(ratings) * 100 if ratings else 0
        print(f"{category}: {count} ({percentage:.1f}%)")
    
    print(f"\nMatch with source ratings: {matches.count(True)}/{len(matches)} ({matches.count(True)/len(matches)*100:.1f}%)")
    
    # Save the results
    if output_file is None:
        output_file = json_file.replace('.json', '_classified.json')
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4)
    print(f"Saved classified articles to {output_file}")
    
    # Also save as CSV
    csv_file = output_file.replace('.json', '.csv')
    pd.DataFrame(articles).to_csv(csv_file, index=False)
    print(f"Saved classified articles to CSV: {csv_file}")
    
    return True


if __name__ == "__main__":
    json_file = "allsides_articles.json" 
    
    # Process all articles (no limit)
    process_articles(json_file, limit=None)
    
    print("\nDone! All articles have been classified.") 