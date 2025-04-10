#!/usr/bin/env python3
"""
Re-classify the cleaned articles with OpenAI model.

This script:
1. Loads the cleaned articles from balanced_dataset/cleaned_articles.json
2. Classifies their political leaning using OpenAI API
3. Saves the classifications back to the file
"""

import json
import os
import sys
import time
import random
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

# OpenAI API configuration - Read from environment variable or set directly
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("Please set it using:")
    print("export OPENAI_API_KEY='your-api-key-here'")
    print("or add it to your .env file and load it with a package like python-dotenv.")
    sys.exit(1)

# Configure the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Define the categories and file paths
CATEGORIES = ["Left", "Lean Left", "Center", "Lean Right", "Right"]
INPUT_FILE = 'balanced_dataset/cleaned_articles.json'
OUTPUT_FILE = 'balanced_dataset/cleaned_articles_rated_openai.json'

def classify_political_leaning(title, full_text, max_tokens=8000):
    """
    Classify the political leaning of an article using OpenAI API.
    Returns a tuple of (numerical_rating, category)
    """
    # Use the full text without truncation
    text_to_analyze = f"{title}\n\n{full_text}"
    
    # Use a numerical rating scale instead of categorical labels
    prompt = f"""Instructions: Political Bias Scale from -6 to 6, where -6 to -3 is left, -3 to -1 is lean-left, -1 to 1 is center, 1 to 3 is lean-right, and 3 to 6 is right. Leaning categories (like lean-left or lean-right) should indicate content that leans towards that side but does not strongly align with it. 

A newspaper article is provided and you have to give it a decimal rating. Only analyze the article's content for language, framing, and overall tone to determine the political bias. Do NOT infer the news outlet or any external context beyond the article itself. If the bias is unclear, output the most appropriate rating based on the overall tone and content.

Article: {text_to_analyze}

Output only a number between -6 and 6 that represents the political rating. No other text.
"""

    try:
        # Make the API call to OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
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
                category = "Left"
            elif rating < -1:
                category = "Lean Left"
            elif rating <= 1:
                category = "Center"
            elif rating < 3:
                category = "Lean Right"
            else:
                category = "Right"
            
            # Return both the numerical rating and the category
            return (rating, category)
            
        except ValueError:
            # If we can't extract a valid number, check for category names in the response
            raw_response_lower = raw_response.lower()
            if "left" in raw_response_lower and "lean" in raw_response_lower:
                return (-2, "Lean Left")
            elif "right" in raw_response_lower and "lean" in raw_response_lower:
                return (2, "Lean Right")
            elif "left" in raw_response_lower:
                return (-4, "Left")
            elif "right" in raw_response_lower:
                return (4, "Right")
            elif "center" in raw_response_lower:
                return (0, "Center")
            else:
                # Default to "No content" if we can't determine the classification
                return (None, "No content")
        
    except Exception as e:
        print(f"Error classifying article: {e}")
        return (None, "Error in classification")

def print_classification_stats(articles):
    """Print statistics about political classifications."""
    # Count classifications
    classification_counts = {category: 0 for category in CATEGORIES}
    ratings = []
    
    for article in articles:
        if 'openai_political_leaning' in article and article['openai_political_leaning'] in CATEGORIES:
            classification_counts[article['openai_political_leaning']] += 1
                
        # Collect numerical ratings for statistics
        if 'openai_political_rating' in article and article['openai_political_rating'] is not None:
            ratings.append(article['openai_political_rating'])
    
    # Print distribution
    print("\nClassification Distribution:")
    total_classified = sum(classification_counts.values())
    for category, count in classification_counts.items():
        percentage = (count / total_classified * 100) if total_classified > 0 else 0
        print(f"{category}: {count} ({percentage:.1f}%)")
    
    # Print numerical rating statistics if available
    if ratings:
        print("\nNumerical Rating Statistics:")
        ratings.sort()
        mean_rating = sum(ratings) / len(ratings)
        median_rating = ratings[len(ratings) // 2]
        print(f"Mean rating: {mean_rating:.2f}")
        print(f"Median rating: {median_rating:.2f}")
        print(f"Rating range: {min(ratings):.1f} to {max(ratings):.1f}")
        
        # Count ratings by category range
        left_count = sum(1 for r in ratings if r <= -3)
        lean_left_count = sum(1 for r in ratings if -3 < r < -1)
        center_count = sum(1 for r in ratings if -1 <= r <= 1)
        lean_right_count = sum(1 for r in ratings if 1 < r < 3)
        right_count = sum(1 for r in ratings if r >= 3)
        
        print("\nNumerical Rating Distribution:")
        print(f"Left range (-6 to -3): {left_count} ({left_count/len(ratings)*100:.1f}%)")
        print(f"Lean Left range (-3 to -1): {lean_left_count} ({lean_left_count/len(ratings)*100:.1f}%)")
        print(f"Center range (-1 to 1): {center_count} ({center_count/len(ratings)*100:.1f}%)")
        print(f"Lean Right range (1 to 3): {lean_right_count} ({lean_right_count/len(ratings)*100:.1f}%)")
        print(f"Right range (3 to 6): {right_count} ({right_count/len(ratings)*100:.1f}%)")

def main():
    # Load the cleaned articles
    print(f"Loading articles from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except Exception as e:
        print(f"Error loading articles: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(articles)} articles")
    
    # Classify each article
    print("\nClassifying articles...")
    for article in tqdm(articles):
        title = article['title']
        full_text = article.get('full_text', "")
        
        # Skip articles without full text
        if not full_text:
            article['openai_political_leaning'] = "No content"
            article['openai_political_rating'] = None
            continue
        
        # Classify the article
        numerical_rating, category = classify_political_leaning(title, full_text)
        
        # Add the classification to the article using different field names
        article['openai_political_rating'] = numerical_rating
        article['openai_political_leaning'] = category
        
        # Add a small delay to avoid rate limiting
        time.sleep(random.uniform(1.0, 2.0))
    
    # Save the results
    print(f"\nSaving results to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4)
    
    # Print statistics
    print_classification_stats(articles)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}") 