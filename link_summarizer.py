from phi_search_agent import search_and_display
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import trafilatura
import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
import os
import time


sambanova_api_key = "540f8914-997e-46c6-829a-ff76f5d4d265"  
os.environ["SAMBANOVA_API_KEY"] = sambanova_api_key
        
llm = ChatSambaNovaCloud(
            model="llama3-70b",
            temperature=0.6,
            max_tokens=4000,
            sambanova_api_key=sambanova_api_key
        )

def extract_text_from_url(url):
    """Extract main text content from a URL."""
    try:
        # Enhanced browser-like headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,/;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Add delay for rate limiting
        time.sleep(2)  # 2 second delay between requests
        
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            # First try to get the content
            response = client.get(url, headers=headers)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            
            # Handle PDFs
            if 'application/pdf' in content_type:
                import io
                import pdfplumber
                
                try:
                    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                        text = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
                        if text:
                            return text
                        return "Error: Could not extract text from PDF"
                except Exception as e:
                    return f"Error: Failed to process PDF - {str(e)}"
            
            # Handle HTML content
            downloaded = response.text
            
            # Special handling for academic sites
            if any(domain in url.lower() for domain in ['researchgate.net', 'sciencedirect.com', 'springer.com', 'academia.edu']):
                # Try to extract main article content
                soup = BeautifulSoup(downloaded, 'html.parser')
                
                # Look for common academic article containers
                article_content = None
                for selector in [
                    'article', 
                    '.article-content',
                    '#article-content',
                    '.paper-content',
                    '.research-content',
                    '.publication-content',
                    '.main-content'
                ]:
                    content = soup.select_one(selector)
                    if content:
                        article_content = content
                        break
                
                if article_content:
                    # Clean the content
                    for tag in article_content.find_all(['script', 'style', 'nav', 'header', 'footer']):
                        tag.decompose()
                    text = article_content.get_text(separator='\n')
                    # Clean up the text
                    lines = (line.strip() for line in text.splitlines())
                    text = '\n'.join(line for line in lines if line)
                    if text:
                        return text
            
            # If special handling didn't work, try trafilatura
            text = trafilatura.extract(downloaded, include_links=False, include_images=False)
            
            # If trafilatura fails, try BeautifulSoup as fallback
            if not text:
                soup = BeautifulSoup(downloaded, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "header", "footer"]):
                    script.decompose()
                text = soup.get_text(separator='\n')
                # Clean up the text
                lines = (line.strip() for line in text.splitlines())
                text = '\n'.join(line for line in lines if line)
            
            if not text:
                return "Error: No content could be extracted from the webpage"
            
            # Limit text length to avoid processing extremely long content
            max_length = 50000  # Adjust this value as needed
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            return text
            
    except httpx.TimeoutException:
        return "Error: Request timed out"
    except httpx.HTTPStatusError as e:
        if e.response.status_code in [403, 401]:
            return f"Error: Access denied (HTTP {e.response.status_code}). This website may require authentication or block automated access."
        elif e.response.status_code in [301, 302, 303, 307, 308]:
            return f"Error: Redirect not followed (HTTP {e.response.status_code})"
        elif e.response.status_code == 429:
            return f"Error: Too many requests (HTTP 429). Please try again later."
        else:
            return f"Error: HTTP {e.response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def filter_documents(texts, embedding_model):
    embeddings_filter = EmbeddingsClusteringFilter(
        embeddings=embedding_model,
        num_clusters=max(1, min(len(texts) // 2, 8)),
        num_closest=max(1, min(len(texts) // 4, 3)),
        threshold=0.85
    )
    filtered_texts = embeddings_filter.transform_documents(texts)
    
    return filtered_texts

def summarize_document(texts, llm, embedding_model):
    filtered_docs = filter_documents(texts, embedding_model)
    
    prompt_template = """
    Provide a comprehensive analysis of the following document in a well-structured format. Your response should follow this structure:

    1. Start with a clear, concise introduction of the main topic (1-2 paragraphs).
    
    2. Follow with "Key Concepts:" as a section heading, then list and explain the fundamental concepts.
    
    3. If applicable, include relevant sections with appropriate headings such as:
       - "Components:" or "Types:"
       - "Applications:"
       - "Advantages:"
       - "Challenges:" or "Limitations:"
    
    4. Each section should use proper markdown formatting:
       - Use # for the main title (if any)
       - Use ## for major section headings
       - Use ### for subsection headings
       - Use bullet points for lists
       - Use **bold** for emphasis
    
    Document content:
    {text}

    Provide your structured analysis:
    """
    
    chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=PromptTemplate(template=prompt_template, input_variables=["text"])
    )
    
    summary = chain.invoke(filtered_docs)
    return summary

def ask_followup_question(content, llm):
    """Handle follow-up questions about the content.""" 
    while True:
        question = input("\nAsk a follow-up question (or 'next' to move to next result, 'quit' to exit): ")
        if question.lower() in ['next', 'quit']:
            return question.lower()
        
        prompt_template = """
        Based on the following content, please answer the question.
        
        Content:
        {text}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["text", "question"])
        
        try:
            # Create a simple chain for question answering
            chain = load_summarize_chain(
                llm,
                chain_type="stuff",
                prompt=prompt
            )
            
            # Create a Document object with the content
            doc = [Document(page_content=content)]
            
            # Get the answer
            response = chain.invoke({"input_documents": doc, "text": content, "question": question})
            print("\nAnswer:")
            print(response['output_text'])
            print("-" * 80)
        except Exception as e:
            print(f"Error processing question: {str(e)}")

def process_search_results():
    """Process search results and generate summaries."""
    try:
        # Initialize models with the specified configurations
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Set up SambaNova API key
        sambanova_api_key = "540f8914-997e-46c6-829a-ff76f5d4d265"  # Replace with your actual API key
        os.environ["SAMBANOVA_API_KEY"] = sambanova_api_key
        
        llm = ChatSambaNovaCloud(
            model="llama3-70b",
            temperature=0.6,
            max_tokens=4000,
            sambanova_api_key=sambanova_api_key
        )
        
        # Get search query
        query = input("Enter your search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            return False
        
        ddgs = DDGS()
        
        results = list(ddgs.text(query, max_results=5))
        
        print("\nProcessing search results for:", query)
        print("-" * 80 + "\n")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Store all content for later use
        all_content = []
        all_texts = []  # Store all document chunks
        
        # First, process and summarize all URLs
        for result in results:
            url = result['href']
            title = result.get('title', 'No title available')
            
            print(f"Analyzing URL: {url}")
            print(f"Title: {title}\n")
            
            content = extract_text_from_url(url)
            
            if content.startswith("Error:"):
                print(f"Could not process URL: {content}\n")
                continue
                
            all_content.append({"url": url, "title": title, "content": content})
            
            # Split content into chunks
            texts = text_splitter.create_documents([content])
            all_texts.extend(texts)
            
            # Generate individual summary
            try:
                summary = summarize_document(texts, llm, embedding_model)
                print(f"Summary of {title}:")
                print("-" * 40)
                print(summary['output_text'].strip())
                print("-" * 80 + "\n")
            except Exception as e:
                print(f"Error generating summary: {str(e)}\n")
        
        if all_texts:
            print("\nGenerating cumulative summary from all sources...")
            print("=" * 80)
            
            try:
                cumulative_summary = summarize_document(all_texts, llm, embedding_model)
                print("\nOverall Summary:")
                print("-" * 40)
                print(cumulative_summary['output_text'].strip())
            except Exception as e:
                print(f"Error generating cumulative summary: {str(e)}")
            
            print("=" * 80 + "\n")
            
            # Handle follow-up questions
            while True:
                action = ask_followup_question("\n".join(doc.page_content for doc in all_texts), llm)
                if action in ['next', 'quit']:
                    break
        
        return True
        
    except Exception as e:
        print(f"Error in process_search_results: {str(e)}")
        return True

if __name__ == "__main__":
    while True:
        should_continue = process_search_results()
        if not should_continue:
            print("Goodbye!")
            break
