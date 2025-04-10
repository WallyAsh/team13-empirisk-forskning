import json
from collections import Counter, defaultdict

# Load data
with open('data/articles_filtered.json', 'r') as f:
    articles = json.load(f)

# Group articles by rating and outlet
outlets_by_rating = defaultdict(list)
for article in articles:
    outlets_by_rating[article['source_rating']].append(article['source_outlet'])

# Count articles per outlet within each rating category
outlet_counts_by_rating = {rating: Counter(outlets) for rating, outlets in outlets_by_rating.items()}

# Find how many outlets are needed to reach 66 articles
print('Outlets needed for 66 articles per category:')
total_outlets = 0

for rating, outlet_counter in outlet_counts_by_rating.items():
    # Sort outlets by number of articles (descending)
    outlets_sorted = sorted(outlet_counter.items(), key=lambda x: x[1], reverse=True)
    
    article_count = 0
    outlets_needed = 0
    outlets_list = []
    
    # Add outlets until we reach 66 articles
    for outlet, count in outlets_sorted:
        article_count += count
        outlets_needed += 1
        outlets_list.append(f"{outlet} ({count})")
        if article_count >= 66:
            break
    
    print(f"\n{rating}: {outlets_needed} outlets needed (total articles: {article_count})")
    print("  " + ", ".join(outlets_list))
    total_outlets += outlets_needed

print(f"\nTotal outlets across all categories: {total_outlets}") 