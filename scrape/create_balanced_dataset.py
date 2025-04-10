import json
import os
import random
from collections import defaultdict

# Load the articles
with open('data/articles_filtered.json', 'r') as f:
    articles = json.load(f)

# Group articles by category and outlet
articles_by_category = defaultdict(list)
for article in articles:
    articles_by_category[article['source_rating']].append(article)

# Create output directory if it doesn't exist
output_dir = 'balanced_dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to select articles with maximum outlet diversity
def select_diverse_articles(category_articles, count=60):
    # Group by outlet
    articles_by_outlet = defaultdict(list)
    for article in category_articles:
        articles_by_outlet[article['source_outlet']].append(article)
    
    # Sort outlets by number of articles (to prioritize outlets with fewer articles)
    outlets_sorted = sorted(articles_by_outlet.items(), key=lambda x: len(x[1]))
    
    selected = []
    # First, take one article from each outlet until we have enough
    for outlet, outlet_articles in outlets_sorted:
        if len(selected) < count:
            selected.append(random.choice(outlet_articles))
        else:
            break
    
    # If we still don't have enough, go back and take more articles from outlets with multiple articles
    if len(selected) < count:
        # Sort outlets by number of articles (descending) to take from outlets with more articles first
        outlets_sorted = sorted(articles_by_outlet.items(), key=lambda x: len(x[1]), reverse=True)
        
        for outlet, outlet_articles in outlets_sorted:
            # Skip if we already took all articles from this outlet
            remaining_articles = [a for a in outlet_articles if a not in selected]
            while remaining_articles and len(selected) < count:
                article = random.choice(remaining_articles)
                selected.append(article)
                remaining_articles.remove(article)
                
            if len(selected) >= count:
                break
    
    return selected[:count]

# Select 60 articles from each category
balanced_dataset = []
category_counts = {}

for category in ["Left", "Lean Left", "Center", "Lean Right", "Right"]:
    if len(articles_by_category[category]) >= 60:
        selected = select_diverse_articles(articles_by_category[category], 60)
        balanced_dataset.extend(selected)
        category_counts[category] = len(selected)
    else:
        # If category has fewer than 60 articles, take all of them
        balanced_dataset.extend(articles_by_category[category])
        category_counts[category] = len(articles_by_category[category])
        print(f"Warning: Only {len(articles_by_category[category])} articles available for {category} category")

# Save the balanced dataset
with open(f'{output_dir}/balanced_articles.json', 'w') as f:
    json.dump(balanced_dataset, f, indent=4)

# Analyze the result
outlet_count = defaultdict(int)
for article in balanced_dataset:
    outlet_count[article['source_outlet']] += 1

print(f"\nCreated balanced dataset with {len(balanced_dataset)} articles:")
for category, count in category_counts.items():
    print(f"  {category}: {count} articles")

print(f"\nNumber of unique source outlets: {len(outlet_count)}")
print("\nTop outlets by number of articles:")
for outlet, count in sorted(outlet_count.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {outlet}: {count} articles") 