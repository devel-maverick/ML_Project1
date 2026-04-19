"""
Fact Checker - searches multiple news sources for related articles
to cross-reference claims. Uses Google News RSS + direct searches
on BBC, Reuters, etc.
"""

import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import quote_plus


# Common words to skip when extracting key terms
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
    'by', 'from', 'as', 'into', 'through', 'during', 'before', 'after',
    'between', 'out', 'off', 'over', 'under', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'not', 'only',
    'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
    'or', 'if', 'while', 'about', 'this', 'that', 'these', 'those', 'it',
    'its', 'his', 'her', 'their', 'our', 'my', 'your', 'he', 'she',
    'they', 'we', 'you', 'said', 'also', 'one', 'two', 'new', 'people',
    'time', 'year', 'first', 'last', 'like', 'get', 'make', 'know',
    'think', 'see', 'say', 'come', 'take', 'going', 'much', 'many',
}

# Trusted news sources with their RSS/search URLs
NEWS_SOURCES = [
    {
        "name": "Google News",
        "url": "https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en",
        "type": "rss"
    },
    {
        "name": "BBC News",
        "url": "https://feeds.bbci.co.uk/news/rss.xml",
        "type": "rss_general"
    },
    {
        "name": "Reuters",
        "url": "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best",
        "type": "rss_general"
    },
]

# Headers to avoid being blocked by news sites
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36'
}


def extract_key_terms(text, max_terms=5):
    """Pick out the most frequent meaningful words from the text."""
    words = re.findall(r'[a-zA-Z]+', text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) >= 4]

    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in sorted_terms[:max_terms]]


def search_google_news(query, max_results=5):
    """Search Google News RSS for related articles."""
    articles = []
    try:
        encoded = quote_plus(query)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, 'xml')
        items = soup.find_all('item')

        for item in items[:max_results]:
            articles.append({
                "title": item.find('title').text.strip() if item.find('title') else "Unknown",
                "url": item.find('link').text.strip() if item.find('link') else "",
                "published": item.find('pubDate').text.strip() if item.find('pubDate') else "Unknown",
                "source": item.find('source').text.strip() if item.find('source') else "Google News"
            })
    except Exception:
        pass
    return articles


def search_bbc_news(query, max_results=3):
    """Search BBC News RSS feed and filter by query terms."""
    articles = []
    try:
        # BBC search RSS
        encoded = quote_plus(query)
        url = f"https://feeds.bbci.co.uk/news/rss.xml"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, 'xml')
        items = soup.find_all('item')

        # Filter items that match any of our search terms
        query_words = set(query.lower().split())
        for item in items:
            title = item.find('title').text.strip() if item.find('title') else ""
            desc = item.find('description').text.strip() if item.find('description') else ""
            combined = (title + " " + desc).lower()

            # Check if any query word appears in the title or description
            if any(word in combined for word in query_words):
                articles.append({
                    "title": title,
                    "url": item.find('link').text.strip() if item.find('link') else "",
                    "published": item.find('pubDate').text.strip() if item.find('pubDate') else "Unknown",
                    "source": "BBC News"
                })
                if len(articles) >= max_results:
                    break
    except Exception:
        pass
    return articles


def search_reuters(query, max_results=3):
    """Try to get Reuters articles via their RSS feed."""
    articles = []
    try:
        url = "https://www.reutersagency.com/feed/?taxonomy=best-sectors&post_type=best"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, 'xml')
        items = soup.find_all('item')

        query_words = set(query.lower().split())
        for item in items:
            title = item.find('title').text.strip() if item.find('title') else ""
            combined = title.lower()

            if any(word in combined for word in query_words):
                articles.append({
                    "title": title,
                    "url": item.find('link').text.strip() if item.find('link') else "",
                    "published": item.find('pubDate').text.strip() if item.find('pubDate') else "Unknown",
                    "source": "Reuters"
                })
                if len(articles) >= max_results:
                    break
    except Exception:
        pass
    return articles


def search_related_articles(text, max_results=8):
    """
    Search multiple news sources for related articles.
    Combines results from Google News, BBC, and Reuters.
    """
    try:
        key_terms = extract_key_terms(text)
        if not key_terms:
            return {
                "query": "", "articles": [],
                "status": "error",
                "message": "Could not extract search terms from article."
            }

        query = " ".join(key_terms[:4])  # Use top 4 terms
        all_articles = []

        # Search all sources
        all_articles.extend(search_google_news(query, max_results=4))
        all_articles.extend(search_bbc_news(query, max_results=2))
        all_articles.extend(search_reuters(query, max_results=2))

        # Remove duplicates by title similarity
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            short_title = article["title"][:50].lower()
            if short_title not in seen_titles:
                seen_titles.add(short_title)
                unique_articles.append(article)

        # Cap at max_results
        unique_articles = unique_articles[:max_results]

        if unique_articles:
            return {
                "query": query,
                "articles": unique_articles,
                "status": "success",
                "message": f"Found {len(unique_articles)} related articles from multiple sources."
            }
        else:
            return {
                "query": query,
                "articles": [],
                "status": "partial",
                "message": f"No matching articles found for: '{query}'. The topic may be too niche or recent."
            }

    except Exception as e:
        return {
            "query": "", "articles": [],
            "status": "error",
            "message": f"Verification failed: {str(e)}"
        }
