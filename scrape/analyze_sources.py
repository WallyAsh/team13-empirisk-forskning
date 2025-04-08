import json
from collections import defaultdict

# Load the data
with open('data/articles_base.json', 'r') as f:
    data = json.load(f)

# Inspect any articles with unusual ratings
print("--- Checking for anomalies in ratings ---")
unusual_ratings = set()
for i, article in enumerate(data):
    if article.get('source_rating') not in ['Left', 'Lean Left', 'Center', 'Lean Right', 'Right']:
        unusual_ratings.add(article.get('source_rating'))
        print(f"Article {i}: Source: {article.get('source_outlet')}, Rating: {article.get('source_rating', 'MISSING')}")
        if len(unusual_ratings) >= 5:  # Show just a few examples
            break

print(f"Unusual ratings found: {unusual_ratings}\n")

# Find sources that appear in multiple rating categories
sources_to_ratings = defaultdict(set)
for article in data:
    sources_to_ratings[article['source_outlet']].add(article.get('source_rating', 'MISSING'))

sources_with_multiple_ratings = {source: ratings for source, ratings in sources_to_ratings.items() if len(ratings) > 1}
print("--- Sources that appear with different ratings ---")
for source, ratings in sources_with_multiple_ratings.items():
    print(f"{source}: {sorted(ratings)}")
print()

# Count articles per rating category
articles_per_category = defaultdict(int)
sources_per_category = defaultdict(set)

for article in data:
    rating = article['source_rating']
    source = article['source_outlet']
    articles_per_category[rating] += 1
    sources_per_category[rating].add(source)

# Print total articles by category
print("\n--- Total Articles by Political Category ---")
for category, count in sorted(articles_per_category.items()):
    print(f"{category}: {count} articles")
print(f"TOTAL: {sum(articles_per_category.values())} articles")

# Print sources by category
print("\n--- Total Sources by Political Category ---")
for category, sources in sorted(sources_per_category.items()):
    print(f"{category}: {len(sources)} sources")
print(f"TOTAL: {len(set().union(*sources_per_category.values()))} unique sources")

# Count sources with 3+ articles in each category
print("\n--- Sources with 3+ Articles by Category ---")
source_counts = defaultdict(int)
for article in data:
    source_counts[article['source_outlet']] += 1

for category, sources in sorted(sources_per_category.items()):
    sources_3plus = [source for source in sources if source_counts[source] >= 3]
    print(f"{category}: {len(sources_3plus)} sources with 3+ articles")
    for source in sorted(sources_3plus):
        print(f"  {source}: {source_counts[source]} articles")

# Count sources with 5+ articles in each category
print("\n--- Sources with 5+ Articles by Category ---")
for category, sources in sorted(sources_per_category.items()):
    sources_5plus = [source for source in sources if source_counts[source] >= 5]
    print(f"{category}: {len(sources_5plus)} sources with 5+ articles")
    for source in sorted(sources_5plus):
        print(f"  {source}: {source_counts[source]} articles") 