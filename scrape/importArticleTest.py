from newspaper import Article

def extract_article_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return "Failed to extract article text"

# Example usage:
original_url = "https://jacobin.com/2025/01/us-foreign-aid-imperialism-humanitarianism"
full_text = extract_article_text(original_url)
print(full_text)
