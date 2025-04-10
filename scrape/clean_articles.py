import json

# Load the articles
with open('data/articles_filtered.json', 'r') as f:
    articles = json.load(f)

# Remove the source_rating_value from each article
for article in articles:
    if 'source_rating_value' in article:
        del article['source_rating_value']

# Save the cleaned articles back to file
with open('data/articles_filtered.json', 'w') as f:
    json.dump(articles, f, indent=4)

print(f"Removed 'source_rating_value' field from {len(articles)} articles") 