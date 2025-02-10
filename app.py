from flask import Flask, request, jsonify
from flask_cors import CORS
from link_summarizer import (
    extract_text_from_url,
    filter_documents,
    summarize_document,
    DDGS
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS to allow requests from your frontend
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize models and configurations
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
sambanova_api_key = "540f8914-997e-46c6-829a-ff76f5d4d265"  
os.environ["SAMBANOVA_API_KEY"] = sambanova_api_key

llm = ChatSambaNovaCloud(
    model="llama3-70b",
    temperature=0.6,
    max_tokens=4000,
    sambanova_api_key=sambanova_api_key
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        logger.info(f"Processing search query: {query}")
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=5))
        
        if not results:
            return jsonify({
                'error': 'No results found for your query',
                'suggestions': [
                    'Try different keywords',
                    'Make sure your search terms are spelled correctly',
                    'Try more general terms'
                ]
            }), 404
        
        processed_results = []
        
        for result in results:
            try:
                url = result['href']
                title = result['title']
                snippet = result.get('body', '')
                
                logger.info(f"Analyzing URL: {url}")
                
                # Extract content from URL
                content = extract_text_from_url(url)
                if content.startswith("Error"):
                    logger.error(f"Could not process URL: {content}")
                    processed_results.append({
                        'url': url,
                        'title': title,
                        'snippet': snippet,
                        'status': 'failed',
                        'error': content
                    })
                    continue
                    
                # Split text into chunks
                texts = text_splitter.create_documents([content])
                
                # Generate summary for this specific URL
                logger.info(f"Generating summary for: {title}")
                summary = summarize_document(texts, llm, embedding_model)
                
                # Extract key points for this summary
                key_points = extract_key_points(summary['output_text'])
                
                # Add to processed results
                processed_results.append({
                    'url': url,
                    'title': title,
                    'snippet': snippet,
                    'status': 'success',
                    'summary': summary['output_text'],
                    'key_points': key_points
                })
                
            except Exception as e:
                logger.error(f"Error processing result: {str(e)}")
                processed_results.append({
                    'url': url,
                    'title': title,
                    'snippet': snippet,
                    'status': 'failed',
                    'error': str(e)
                })
                continue

        # Generate cumulative summary
        cumulative_summary = None
        try:
            if texts:
                logger.info("Generating cumulative summary...")
                cumulative_summary = summarize_document(texts, llm, embedding_model)
        except Exception as e:
            logger.error(f"Error generating cumulative summary: {str(e)}")

        # Organize the response
        response = {
            'query': query,
            'cumulative_summary': cumulative_summary['output_text'] if cumulative_summary else None,
            'sources': processed_results,
            'metadata': {
                'total_sources': len(results),
                'successful_summaries': len([r for r in processed_results if r['status'] == 'success']),
                'failed_summaries': len([r for r in processed_results if r['status'] == 'failed']),
                'timestamp': datetime.now().isoformat()
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e),
            'suggestions': [
                'Please try again later',
                'Your query might be too complex',
                'Try a different search term'
            ]
        }), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question')
        content = data.get('content')
        
        if not question or not content:
            return jsonify({'error': 'Question and content are required'}), 400

        logger.info(f"Processing follow-up question: {question}")

        prompt_template = """
        Based on the following content, please provide a detailed answer to the question.
        Include specific references to the content where relevant.
        
        Content:
        {text}
        
        Question: {question}
        
        Please structure your answer with:
        1. Direct answer to the question
        2. Supporting evidence from the content
        3. Any relevant context or caveats
        
        Answer:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["text", "question"])
        
        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=prompt
        )
        
        doc = [Document(page_content=content)]
        response = chain.invoke({"input_documents": doc, "text": content, "question": question})

        # Structure the response
        answer_response = {
            'answer': response['output_text'],
            'metadata': {
                'question': question,
                'timestamp': datetime.datetime.now().isoformat(),
                'confidence': 'high' if len(response['output_text']) > 100 else 'medium'
            },
            'suggestions': generate_follow_up_suggestions(question, response['output_text'])
        }

        logger.info("Question processed successfully")
        return jsonify(answer_response)

    except Exception as e:
        logger.error(f"Error in ask_question endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'message': 'An error occurred while processing your question'
        }), 500

def extract_key_points(text):
    """Extract key points from the summary text."""
    # Split the text into sentences and look for key points
    sentences = text.split('. ')
    key_points = []
    
    for sentence in sentences:
        # Look for sentences that are likely to be key points
        if any(indicator in sentence.lower() for indicator in 
               ['important', 'key', 'significant', 'main', 'crucial', 'essential',
                'finding', 'conclusion', 'result', 'shows', 'demonstrates']):
            key_points.append(sentence.strip())
    
    return key_points[:5]  # Return top 5 key points

def generate_follow_up_suggestions(question, answer):
    """Generate suggested follow-up questions based on the current Q&A."""
    # Simple logic to generate follow-up questions
    suggestions = [
        f"Can you elaborate more on specific aspects of this topic?",
        f"What are the implications of these findings?",
        f"How does this compare to other similar cases or studies?",
        f"What are the potential limitations or challenges in this context?"
    ]
    return suggestions

if __name__ == '__main__':
    app.run(debug=True, port=5000)