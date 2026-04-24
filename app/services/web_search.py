import requests
from bs4 import BeautifulSoup
from googlesearch import search as google_search
from urllib.parse import quote_plus
import time
from app.config import Config

def web_search_fallback(query, num_results=Config.WEB_SEARCH_RESULTS):
    """Perform web search as fallback when local context is insufficient"""
    try:
        search_results = []
        
        # Perform Google search
        try:
            search_iterator = google_search(query, num_results=num_results, advanced=True)
            urls = []
            for result in search_iterator:
                if hasattr(result, 'url'):
                    urls.append(result.url)
                else:
                    urls.append(str(result))
        except Exception as e:
            print(f"Google search failed: {str(e)}")
            search_query = quote_plus(query)
            return search_results
        
        # Fetch content from each URL
        for url in urls[:num_results]:
            try:
                # Skip if it's not a valid URL
                if not url.startswith('http'):
                    continue
                    
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.find('title')
                title_text = title.get_text() if title else 'No title'
                
                # Extract main content                                                  
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                if main_content:
                    text = ' '.join([p.get_text() for p in main_content.find_all('p')])
                else:
                    # Fallback: get all paragraphs
                    text = ' '.join([p.get_text() for p in soup.find_all('p')])
                
                text = text[:2000]  # Limit length
                
                search_results.append({
                    "title": title_text,
                    "url": url,
                    "content": text
                })
                
                time.sleep(1)  # Be polite to servers
                
            except Exception as e:
                print(f"Error fetching {url}: {str(e)}")
                continue
        
        return search_results
        
    except Exception as e:
        print(f"Web search failed: {str(e)}")
        return []

def summarize_web_results(results, max_chars_each=500, max_results=3):
    """Return a compact summary of top web results (title, url, short snippet)."""
    if not results:
        return "No web results found."

    items = []
    for r in results[:max_results]:
        title = r.get('title', 'No title')
        url = r.get('url', 'No URL')
        content = (r.get('content') or '').strip().replace('\n', ' ')
        snippet = content[:max_chars_each].rsplit(' ', 1)[0] + ('...' if len(content) > max_chars_each else '')
        items.append(f"{title}\n{url}\n{snippet}")

    return "\n\n".join(items)

def format_web_results(results):
    """Format web search results for LLM context (condensed)."""
    return summarize_web_results(results, max_chars_each=500, max_results=3)
