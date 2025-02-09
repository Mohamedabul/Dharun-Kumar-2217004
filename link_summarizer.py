# from backend.phi_search_agent import search_and_display
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import trafilatura
import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_transformers import EmbeddingsClusteringFilter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
import os

def extract_text_from_url(url):
    """Extract main text content from a URL."""
    try:
        # Download with httpx and timeout
        with httpx.Client() as client:
            response = client.get(url, timeout=20)
            downloaded = response.text
            # Extract text content
            text = trafilatura.extract(downloaded, include_links=False, include_images=False)
            return text if text else "Unable to extract content"
    except Exception as e:
        return f"Error accessing URL: {str(e)}"
    
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
    Please provide a comprehensive summary of the following document. Focus on:
    1. Summary of the Source
    2. Main themes and key points
    3. Important findings or conclusions

    Document content:
    {text}

    Summary:
    """
    
    chain = load_summarize_chain(
        llm,
        chain_type="stuff",
        prompt=PromptTemplate(template=prompt_template, input_variables=["text"])
    )
    
    summary = chain.invoke(filtered_docs)
    print("\nComprehensive Summary:")
    print(summary['output_text'])
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
    # Initialize models with the specified configurations
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
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
    print("-" * 80)
    
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
        print(f"\nAnalyzing URL: {url}")
        print(f"Title: {result['title']}")
        
        # Extract content from URL
        content = extract_text_from_url(url)
        if content.startswith("Error"):
            print(f"Could not process URL: {content}")
            continue
            
        # Split text into chunks
        texts = text_splitter.create_documents([content])
        
        try:
            # Generate summary using the clustering and LLM approach
            print(f"\nGenerating summary for: {result['title']}")
            summary = summarize_document(texts, llm, embedding_model)
            all_content.append(content)
            all_texts.extend(texts)  # Add document chunks to all_texts
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            continue
            
        print("-" * 80)
    
    # Generate cumulative summary from all texts
    if all_texts:
        print("\nGenerating cumulative summary from all sources...")
        print("=" * 80)
        cumulative_summary = summarize_document(all_texts, llm, embedding_model)
        print("=" * 80)
    
    # After all summaries, handle follow-up questions about all content
    combined_content = "\n\n".join(all_content)
    while True:
        user_choice = ask_followup_question(combined_content, llm)
        if user_choice == 'quit':
            return False
        if user_choice == 'next':
            break
    
    return True

if __name__ == "__main__":
    while True:
        should_continue = process_search_results()
        if not should_continue:
            print("Goodbye!")
            break 