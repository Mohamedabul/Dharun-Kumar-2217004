from duckduckgo_search import DDGS
import httpx

def search_and_display():
    while True:
        query = input("Enter your search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        ddgs = DDGS(
            timeout=20,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        )
        
        results = list(ddgs.text(query))
            
        print("\nSearch Results for:", query)
        print("-" * 80)
        
        for result in results:
            print(f"\nTitle: {result['title']}")
            print(f"Link: {result['href']}")
            print(f"Summary: {result['body']}")
            print("-" * 80)

if __name__ == "__main__":
    search_and_display()
