import json
from collections import Counter, defaultdict

# Load the data
with open('data/articles_filtered.json', 'r') as f:
    articles = json.load(f)

# Count articles by outlet
outlet_counts = Counter([article['source_outlet'] for article in articles])

# Get outlets with 3+ articles
outlets_with_3plus = {outlet: count for outlet, count in outlet_counts.items() if count >= 3}

# Create a mapping from outlet to rating
outlet_to_rating = {}
for article in articles:
    outlet = article['source_outlet']
    if outlet not in outlet_to_rating and outlet in outlets_with_3plus:
        outlet_to_rating[outlet] = article['source_rating']

# Count outlets with 3+ articles by rating
ratings = defaultdict(int)
for outlet, rating in outlet_to_rating.items():
    ratings[rating] += 1

# Display results
print("Number of outlets with 3+ articles by rating:")
for rating, count in ratings.items():
    print(f"{rating}: {count}")

print("\nDetailed list of outlets with 3+ articles by rating:")
outlets_by_rating = defaultdict(list)
for outlet, rating in outlet_to_rating.items():
    outlets_by_rating[rating].append((outlet, outlet_counts[outlet]))

for rating, outlets in outlets_by_rating.items():
    print(f"\n{rating}:")
    for outlet, count in sorted(outlets, key=lambda x: x[1], reverse=True):
        print(f"  - {outlet}: {count} articles") 