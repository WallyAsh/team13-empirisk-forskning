import json
from collections import defaultdict

# Load the filtered data
print("Loading filtered articles data...")
with open('data/articles_filtered.json', 'r') as f:
    data = json.load(f)

print(f"Dataset size: {len(data)} articles")

# Count articles per rating category
articles_per_category = defaultdict(int)
sources_per_category = defaultdict(set)

for article in data:
    rating = article['source_rating']
    source = article['source_outlet']
    articles_per_category[rating] += 1
    sources_per_category[rating].add(source)

# Print total articles by category
print("\n--- Articles by Political Category ---")
for category, count in sorted(articles_per_category.items()):
    print(f"{category}: {count} articles ({count/len(data)*100:.1f}%)")
print(f"TOTAL: {sum(articles_per_category.values())} articles")

# Print sources by category
print("\n--- Sources by Political Category ---")
for category, sources in sorted(sources_per_category.items()):
    print(f"{category}: {len(sources)} sources")
print(f"TOTAL: {len(set().union(*sources_per_category.values()))} unique sources")

# Count sources with 5+ articles in each category
print("\n--- Sources with 5+ Articles by Category ---")
source_counts = defaultdict(int)
for article in data:
    source_counts[article['source_outlet']] += 1

for category, sources in sorted(sources_per_category.items()):
    sources_5plus = [source for source in sources if source_counts[source] >= 5]
    print(f"{category}: {len(sources_5plus)} sources with 5+ articles")
    for source in sorted(sources_5plus):
        print(f"  {source}: {source_counts[source]} articles") 