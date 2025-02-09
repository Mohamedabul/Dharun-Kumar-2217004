from flask import Flask, request, jsonify
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

app = Flask(__name__)

# Initialize models and configurations
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
sambanova_api_key = "540f8914-997e-46c6-829a-ff76f5d4d265"  # Replace with your actual API key
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

        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=5))
        
        all_content = []
        all_texts = []
        summaries = []

        for result in results:
            url = result['href']
            title = result['title']
            
            # Extract content from URL
            content = extract_text_from_url(url)
            if content.startswith("Error"):
                continue
                
            # Split text into chunks
            texts = text_splitter.create_documents([content])
            
            try:
                # Generate summary
                summary = summarize_document(texts, llm, embedding_model)
                summaries.append({
                    'url': url,
                    'title': title,
                    'summary': summary['output_text']
                })
                all_content.append(content)
                all_texts.extend(texts)
                
            except Exception as e:
                continue

        # Generate cumulative summary
        cumulative_summary = None
        if all_texts:
            cumulative_summary = summarize_document(all_texts, llm, embedding_model)

        response = {
            'individual_summaries': summaries,
            'cumulative_summary': cumulative_summary['output_text'] if cumulative_summary else None
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question')
        content = data.get('content')
        
        if not question or not content:
            return jsonify({'error': 'Question and content are required'}), 400

        from langchain.prompts import PromptTemplate
        from langchain.chains.summarize import load_summarize_chain
        from langchain.docstore.document import Document

        prompt_template = """
        Based on the following content, please answer the question.
        
        Content:
        {text}
        
        Question: {question}
        
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

        return jsonify({'answer': response['output_text']})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 