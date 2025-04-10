import json
from collections import Counter

# Load the balanced dataset
with open('balanced_dataset/balanced_articles.json', 'r') as f:
    articles = json.load(f)

# Count articles by category
category_counts = Counter(article['source_rating'] for article in articles)

# Print results
print(f"Total articles in balanced dataset: {len(articles)}")
print("\nArticles by category:")
for category in ["Left", "Lean Left", "Center", "Lean Right", "Right"]:
    count = category_counts.get(category, 0)
    print(f"  {category}: {count} articles")
    
    if count != 60:
        print(f"    WARNING: Expected 60 articles, found {count}")

# Count articles by outlet
outlet_counts = Counter(article['source_outlet'] for article in articles)
print(f"\nNumber of unique outlets: {len(outlet_counts)}")

# Show top outlets
print("\nTop 5 outlets:")
for outlet, count in outlet_counts.most_common(5):
    print(f"  {outlet}: {count} articles") 