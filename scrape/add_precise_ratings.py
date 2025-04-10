import json

# Load the articles
with open('data/articles_filtered.json', 'r') as f:
    articles = json.load(f)

# Load the ratings cache
with open('data/allsides_ratings_cache.json', 'r') as f:
    ratings_cache = json.load(f)

# Count how many articles were updated
updated_count = 0
missing_count = 0
missing_sources = set()

# Add precise rating values to each article
for article in articles:
    source_url = article.get('source_url')
    if source_url in ratings_cache:
        article['source_rating_value_precise'] = ratings_cache[source_url]
        updated_count += 1
    else:
        article['source_rating_value_precise'] = None
        missing_count += 1
        missing_sources.add(article['source_outlet'])

# Save the updated articles back to file
with open('data/articles_filtered.json', 'w') as f:
    json.dump(articles, f, indent=4)

print(f"Added precise rating values to {updated_count} articles")
if missing_count > 0:
    print(f"Warning: Could not find precise ratings for {missing_count} articles (set to null)")
    print(f"Missing sources: {', '.join(sorted(missing_sources))}") 