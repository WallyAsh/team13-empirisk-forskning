import json
import re
import os

# Load the balanced dataset
with open('balanced_dataset/balanced_articles.json', 'r') as f:
    articles = json.load(f)

# Get all source outlet names for pattern matching
source_outlets = set(article['source_outlet'] for article in articles)
# Add variations of outlet names (e.g., "New York Times" -> "NY Times", "NYT")
outlet_variations = {
    "New York Times": ["NY Times", "NYT"],
    "Washington Post": ["WaPo", "Washington Post's"],
    "CNN": ["CNN's", "Cable News Network"],
    "Associated Press": ["AP", "AP's", "Associated Press'"],
    "Fox News": ["Fox", "Fox's", "FOX"],
    "BBC": ["BBC's", "BBC News"],
    "Reuters": ["Reuters'"],
    "NPR": ["National Public Radio"],
    "CNBC": ["CNBC's"],
    "MSNBC": ["MSNBC's"],
    "USA TODAY": ["USA Today", "USA Today's"],
    "Wall Street Journal": ["WSJ", "The Journal"],
    "Newsweek": ["Newsweek's"],
    "Politico": ["Politico's"],
    "The Guardian": ["Guardian"],
    "The Hill": ["The Hill's"],
    "The Atlantic": ["The Atlantic's"],
    "HuffPost": ["Huffington Post"],
    "Washington Examiner": ["Examiner"],
    "Washington Times": ["Times"],
    "New York Post": ["NY Post"],
    "Vox": ["Vox's"],
    "Bloomberg": ["Bloomberg News"],
    "CBS News": ["CBS"],
    "NBC News": ["NBC"],
    "ABC News": ["ABC"]
}

all_outlet_patterns = list(source_outlets)
for outlet, variations in outlet_variations.items():
    all_outlet_patterns.extend(variations)

# Create regex patterns for cleaning
outlet_pattern = r'\b(' + '|'.join(re.escape(outlet) for outlet in all_outlet_patterns) + r')\b'
reporter_pattern = r'(?:By|Reporting by|reported by|Written by)\s+[A-Z][a-z]+\s+[A-Z][a-z]+'
byline_pattern = r'(?:By|Reporting by).*?(?:\n|\.)'
email_pattern = r'\S+@\S+\.\S+'
citation_pattern = r'(?:according to|reported by|told|said)\s+(?:the\s+)?(?:' + outlet_pattern + r')'
reuters_photo_pattern = r'REUTERS/[^\s,\.]+'
edit_by_pattern = r'(?:edited|Edited) by\s+.+?(?:;|\.|$)'
link_pattern = r'https?://\S+'
location_date_pattern = r'[A-Z]+(?:/[A-Z]+)?, [A-Za-z]+ \d+\s+\([^)]+\)\s*-'
category_pattern = r'(?:category|Editor\'s Picks|EDITOR\'S PICK)[^\n\.]+'

# Function to clean a single article's text
def clean_article_text(text):
    # Replace specific outlet name patterns
    text = re.sub(outlet_pattern, "[NEWS OUTLET]", text, flags=re.IGNORECASE)
    
    # Replace reporter bylines
    text = re.sub(reporter_pattern, "", text)
    text = re.sub(byline_pattern, "", text)
    
    # Replace editor information
    text = re.sub(edit_by_pattern, "", text)
    
    # Replace email addresses
    text = re.sub(email_pattern, "[EMAIL]", text)
    
    # Replace citations to specific outlets
    text = re.sub(citation_pattern, "according to a news source", text, flags=re.IGNORECASE)
    
    # Replace Reuters photo credits
    text = re.sub(reuters_photo_pattern, "[PHOTO CREDIT]", text)
    
    # Replace links
    text = re.sub(link_pattern, "[LINK]", text)
    
    # Replace location/date datelines at the beginning of articles
    text = re.sub(location_date_pattern, "", text)
    
    # Replace category tags often at end of articles
    text = re.sub(category_pattern, "", text, flags=re.IGNORECASE)
    
    # Additional cleaning for specific patterns found in articles
    text = re.sub(r'\. ,', ".", text)  # Fix spacing after replacements
    text = re.sub(r'X / Editor\'s Picks', "", text)
    text = re.sub(r'X /', "", text)
    text = re.sub(r'Email.*?Editor\'s Picks', "", text)
    text = re.sub(r'The previous shell command ended, so on the next invocation of this tool, you will be reusing the shell.', "", text)
    
    # Cleaning up paragraph breaks and double spaces
    text = re.sub(r'\n+', "\n\n", text)
    text = re.sub(r' +', " ", text)
    
    return text.strip()

# Clean all articles
cleaned_count = 0
for article in articles:
    original_text = article['full_text']
    cleaned_text = clean_article_text(original_text)
    
    if original_text != cleaned_text:
        cleaned_count += 1
    
    article['full_text'] = cleaned_text

# Save the cleaned dataset
output_dir = 'balanced_dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(f'{output_dir}/cleaned_articles.json', 'w') as f:
    json.dump(articles, f, indent=4)

print(f"Cleaned {cleaned_count} articles out of {len(articles)}")
print(f"Saved to {output_dir}/cleaned_articles.json") 