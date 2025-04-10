#!/usr/bin/env python3
"""
Re-classify the cleaned articles with Google's Gemini model.

This script:
1. Loads the cleaned articles from top5_per_source_final.json
2. Classifies their political leaning using Gemini 2.0 Flash API
3. Saves the classifications back to the file
"""

import json
import os
import sys
import time
import random
from tqdm import tqdm
import threading
from collections import deque
from datetime import datetime, timedelta

# Try to load .env file if dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file if it exists
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed. Using environment variables directly.")
    print("To use a .env file, install python-dotenv: pip install python-dotenv")

# Check if google-genai is installed
try:
    from google import genai
except ImportError:
    print("Error: google-genai package is not installed.")
    print("Please install it using: pip install -q -U google-genai")
    sys.exit(1)

# Gemini API configuration - Read from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable is not set.")
    print("Please set it using:")
    print("export GEMINI_API_KEY='your-api-key-here'")
    print("or add it to your .env file and load it with a package like python-dotenv.")
    sys.exit(1)

# Configure the Gemini client
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Rate limiter class for Gemini API
class RateLimiter:
    def __init__(self, max_requests_per_minute=10):
        self.max_requests = max_requests_per_minute
        self.request_times = deque()
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if we're about to exceed our rate limit"""
        with self.lock:
            now = datetime.now()
            
            # Remove request timestamps older than 1 minute
            while self.request_times and self.request_times[0] < now - timedelta(minutes=1):
                self.request_times.popleft()
            
            # If we've hit our limit for the last minute, wait until we can make another request
            if len(self.request_times) >= self.max_requests:
                wait_time = (self.request_times[0] + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    print(f"\nRate limit approached. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time + 0.1)  # Add a small buffer
            
            # Add this request's timestamp
            self.request_times.append(now)

# Create rate limiter with safer limit (10 requests per minute)
rate_limiter = RateLimiter(max_requests_per_minute=5)

# Define the categories and file paths
CATEGORIES = ["Left", "Lean Left", "Center", "Lean Right", "Right"]
INPUT_FILE = 'best_articles/top5_per_source_final.json'
OUTPUT_FILE = 'best_articles/top5_per_source_rated_gemini.json'

def classify_political_leaning(title, full_text, max_length=30000):
    """
    Classify the political leaning of an article using Gemini API.
    Returns a tuple of (numerical_rating, category)
    """
    # Wait if needed to avoid rate limiting
    rate_limiter.wait_if_needed()
    
    # Truncate the text if it's too long (Gemini might have input limits)
    if full_text and len(full_text) > max_length:
        text_sample = full_text[:max_length]
        text_to_analyze = f"{title}\n\nExcerpt from article:\n{text_sample}..."
    else:
        text_to_analyze = f"{title}\n\n{full_text}"
    
    # Create the prompt
    prompt = f"""Instructions: Political Bias Scale from -6 to 6, where -6 to -3 is left, -3 to -1 is lean-left, -1 to 1 is center, 1 to 3 is lean-right, and 3 to 6 is right. Leaning categories (like lean-left or lean-right) should indicate content that leans towards that side but does not strongly align with it. 

A newspaper article is provided and you have to give it a decimal rating. Only analyze the article's content for language, framing, and overall tone to determine the political bias. Do NOT infer the news outlet or any external context beyond the article itself. If the bias is unclear, output the most appropriate rating based on the overall tone and content.

Article: {text_to_analyze}

Output only a number between -6 and 6 that represents the political rating. No other text.
"""

    try:
        # Make the API call to Gemini
        response = genai_client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            contents=prompt
        )
        
        # Extract the response text
        raw_response = response.text.strip()
        
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
        if 'gemini_political_leaning' in article and article['gemini_political_leaning'] in CATEGORIES:
            classification_counts[article['gemini_political_leaning']] += 1
                
        # Collect numerical ratings for statistics
        if 'gemini_political_rating' in article and article['gemini_political_rating'] is not None:
            ratings.append(article['gemini_political_rating'])
    
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
            article['gemini_political_leaning'] = "No content"
            article['gemini_political_rating'] = None
            continue
        
        # Classify the article
        numerical_rating, category = classify_political_leaning(title, full_text)
        
        # Add the classification to the article using different field names
        article['gemini_political_rating'] = numerical_rating
        article['gemini_political_leaning'] = category
        
        # Add a small delay for safety, but main rate limiting is handled by RateLimiter
        time.sleep(random.uniform(0.5, 1.0))
    
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