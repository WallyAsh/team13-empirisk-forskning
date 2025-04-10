import json
from collections import Counter

# Load the data
with open('data/articles_filtered.json', 'r') as f:
    articles = json.load(f)

# 1. Total number of articles
total_articles = len(articles)
print(f"Total articles: {total_articles}")

# 2. Number of unique source outlets
source_outlets = set(article['source_outlet'] for article in articles)
print(f"Number of unique source outlets: {len(source_outlets)}")
print("\nTop 10 source outlets by article count:")
source_count = Counter(article['source_outlet'] for article in articles)
for outlet, count in source_count.most_common(10):
    print(f"  {outlet}: {count} articles")

# 3. Count articles by category
rating_count = Counter(article['source_rating'] for article in articles)
print("\nArticles by category:")
for rating in ["Left", "Lean Left", "Center", "Lean Right", "Right"]:
    print(f"  {rating}: {rating_count[rating]} articles")

# 4. Count articles with precise ratings
precise_rating_count = sum(1 for article in articles if article['source_rating_value_precise'] is not None)
print(f"\nArticles with precise ratings: {precise_rating_count}")
print(f"Articles without precise ratings: {total_articles - precise_rating_count}")

# 5. Range of precise ratings
precise_ratings = [article['source_rating_value_precise'] for article in articles if article['source_rating_value_precise'] is not None]
if precise_ratings:
    print(f"Range of precise ratings: {min(precise_ratings)} to {max(precise_ratings)}") 