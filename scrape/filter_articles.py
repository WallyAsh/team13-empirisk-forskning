import json
import os

# Load the original data
print("Loading original articles data...")
with open('data/articles_base.json', 'r') as f:
    data = json.load(f)

print(f"Original dataset size: {len(data)} articles")

# Filter out articles with empty or "Not available" text
filtered_data = []
removed = 0

for article in data:
    full_text = article.get('full_text', '')
    
    # Skip articles with empty text or "Not available"
    if not full_text or full_text.strip().lower().startswith("not available"):
        removed += 1
        continue
    
    filtered_data.append(article)

print(f"Removed {removed} articles with empty or 'Not available' text")
print(f"New dataset size: {len(filtered_data)} articles")

# Save the filtered data to a new file
output_path = 'data/articles_filtered.json'
with open(output_path, 'w') as f:
    json.dump(filtered_data, f, indent=4)

print(f"Filtered articles saved to {output_path}")

# Also count articles by category in the filtered data
articles_per_category = {}
for article in filtered_data:
    rating = article['source_rating']
    articles_per_category[rating] = articles_per_category.get(rating, 0) + 1

print("\nFiltered articles distribution:")
for category, count in sorted(articles_per_category.items()):
    print(f"{category}: {count} articles") 