from flask import Flask, request, jsonify
from flask_cors import CORS
from link_summarizer import (
    extract_text_from_url,
    filter_documents,
    summarize_document,
    DDGS,llm,PromptTemplate,load_summarize_chain
)
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import logging
from datetime import datetime
from chat_agent import ChatAgent
import asyncio
from functools import wraps
from langchain_community.chat_models.sambanova import ChatSambaNovaCloud
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS to allow requests from your frontend
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": True
    }
})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Initialize models and configurations
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Initialize chat agent
chat_agent = ChatAgent()

def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        logger.info(f"Processing search query: {query}")
        ddgs = DDGS()
        
        # Get more results initially to account for potential failures
        results = list(ddgs.text(query, max_results=10))
        
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
        successful_texts = []
        required_summaries = 5
        
        for result in results:
            if len([r for r in processed_results if r['status'] == 'success']) >= required_summaries:
                break
                
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
                    # Try next URL if this one failed
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
                
                successful_texts.extend(texts)
                
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

        # If we don't have enough successful summaries, try to get more results
        if len([r for r in processed_results if r['status'] == 'success']) < required_summaries:
            try:
                # Try to get more results
                more_results = list(ddgs.text(query, max_results=5))
                for result in more_results[5:]:  # Start from where we left off
                    if len([r for r in processed_results if r['status'] == 'success']) >= required_summaries:
                        break
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
                        
                        successful_texts.extend(texts)
                        
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
            except Exception as e:
                logger.error(f"Error getting additional results: {str(e)}")

        # Generate cumulative summary if we have successful texts
        cumulative_summary = None
        if successful_texts:
            try:
                logger.info("Generating cumulative summary...")
                cumulative_summary = summarize_document(successful_texts, llm, embedding_model)
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

@app.route('/chat', methods=['POST', 'OPTIONS'])
@async_route
async def chat():
    """
    Handle general chat conversations with the AI.
    """
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.get_json()
        message = data.get('message')
        context = data.get('context', {})

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Process the message using the chat agent
        response = await chat_agent.process_message(message, context)
        
        if not response['success']:
            return jsonify({
                'error': 'Failed to process message',
                'details': response.get('error', 'Unknown error')
            }), 500
        
        # Generate suggestions for follow-up questions
        suggestions = await chat_agent.generate_suggestions(message)
        
        # Add suggestions to the response
        response['suggestions'] = suggestions

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e)
        }), 500

@app.route('/chat/history', methods=['GET'])
def get_chat_history():
    """
    Get the current chat history.
    """
    try:
        history = chat_agent.get_chat_history()
        return jsonify({'history': history})
    except Exception as e:
        app.logger.error(f"Error getting chat history: {str(e)}")
        return jsonify({
            'error': 'An error occurred while retrieving chat history',
            'details': str(e)
        }), 500

@app.route('/chat/clear', methods=['POST'])
def clear_chat_history():
    """
    Clear the chat history.
    """
    try:
        chat_agent.clear_history()
        return jsonify({'message': 'Chat history cleared successfully'})
    except Exception as e:
        app.logger.error(f"Error clearing chat history: {str(e)}")
        return jsonify({
            'error': 'An error occurred while clearing chat history',
            'details': str(e)
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