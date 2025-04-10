import json
from collections import Counter

# Load the balanced dataset
with open('balanced_dataset/balanced_articles.json', 'r') as f:
    articles = json.load(f)

# Get all unique source outlets
source_outlets = set(article['source_outlet'] for article in articles)

# Count by category
outlets_by_category = {}
for category in ["Left", "Lean Left", "Center", "Lean Right", "Right"]:
    category_articles = [a for a in articles if a['source_rating'] == category]
    category_outlets = set(a['source_outlet'] for a in category_articles)
    outlets_by_category[category] = category_outlets

# Print results
print(f"Total number of unique source outlets in balanced dataset: {len(source_outlets)}")
print("\nUnique outlets by category:")
for category, outlets in outlets_by_category.items():
    print(f"  {category}: {len(outlets)} outlets")

# List all outlets
print("\nAll unique source outlets:")
for outlet in sorted(source_outlets):
    categories = []
    for category, outlets in outlets_by_category.items():
        if outlet in outlets:
            categories.append(category)
    print(f"  {outlet} (appears in: {', '.join(categories)})") 