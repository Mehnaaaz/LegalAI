from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pdfplumber
import docx2txt
import numpy as np
from sentence_transformers import SentenceTransformer
import tempfile
import logging
from together import Together
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize embedding model for semantic search
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Together AI client
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY environment variable is not set")
client = Together(api_key=TOGETHER_API_KEY)

# Global variables to store document data
document_chunks = []
document_embeddings = None

def save_file(file):
    """Save the uploaded file securely to a temporary location."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
            file.save(tmp_file.name)
            logger.info(f"File saved to {tmp_file.name}")
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise

def extract_text(file_path):
    """Extract text from supported file types (PDF, DOCX, TXT)."""
    try:
        if file_path.endswith('.pdf'):
            with pdfplumber.open(file_path) as pdf:
                text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
                logger.info(f"Extracted text from PDF: {file_path}")
                return text
        elif file_path.endswith('.docx'):
            text = docx2txt.process(file_path)
            logger.info(f"Extracted text from DOCX: {file_path}")
            return text
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                logger.info(f"Extracted text from TXT: {file_path}")
                return text
        logger.warning(f"Unsupported file type: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return None

def chunk_text(text, chunk_size=500):
    """Split text into chunks of specified word size."""
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and process them into chunks and embeddings."""
    global document_chunks, document_embeddings
    
    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        logger.error("No file selected")
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save and extract text from the file
        file_path = save_file(file)
        text = extract_text(file_path)
        
        try:
            os.remove(file_path)
            logger.info(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {file_path}: {str(e)}")
        
        if not text:
            logger.error(f"Text extraction failed or returned empty for {file_path}")
            return jsonify({'error': 'Unsupported file type or empty file'}), 400
            
        # Process text into chunks and generate embeddings
        document_chunks = chunk_text(text)
        document_embeddings = embedding_model.encode(document_chunks)
        logger.info("Embeddings generated successfully")
        
        return jsonify({
            'message': 'File uploaded and processed successfully',
            'chunk_count': len(document_chunks)
        })
    
    except Exception as e:
        logger.error(f"Upload processing failed: {str(e)}")
        return jsonify({'error': f"Failed to process document: {str(e)}"}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer a question based on the uploaded document using Together AI."""
    global document_chunks, document_embeddings
    
    if not document_chunks:
        logger.error("No document uploaded yet")
        return jsonify({'error': 'No document uploaded yet'}), 400
    
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        logger.error("Question cannot be empty")
        return jsonify({'error': 'Question cannot be empty'}), 400

    try:
        # Perform semantic search to find relevant chunks
        question_embedding = embedding_model.encode(question)
        similarities = np.dot(document_embeddings, question_embedding)
        top_indices = np.argsort(similarities)[-5:][::-1]  # Get top 5 most relevant chunks
        relevant_indices = [i for i in top_indices if similarities[i] > 0.5]
        if not relevant_indices:
            relevant_indices = top_indices[:1]  # Fallback to top 1 if none above threshold
        relevant_chunks = [document_chunks[i] for i in relevant_indices]
        context = "\n\n".join(relevant_chunks)
        
        # Prepare messages for the Together AI API
        messages = [
            {"role": "system", "content": "You are a legal assistant AI. Provide accurate, concise answers in 2-3 sentences based solely on the given context. Ensure the response directly addresses the question, even for summaries or explanations."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
        
        # Call the Together AI API with Llama-2-7b-chat-hf
        response = client.chat.completions.create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["[/INST]", "</s>"],
            stream=False
        )
        
        # Log the full response for debugging
        logger.info(f"API Response: {response.choices[0].message.content}")
        logger.info(f"Finish Reason: {response.choices[0].finish_reason}")
        
        # Extract the answer and validate
        answer = response.choices[0].message.content.strip()
        if not answer or len(answer) < 10:
            logger.warning("Answer too short or empty")
            answer = "Sorry, I couldn’t generate a meaningful answer from the context. Please try rephrasing your question or provide more context."
        
        # Return the answer and relevant chunks
        return jsonify({
            'answer': answer,
            'relevant_chunks': relevant_chunks
        })
    
    except Exception as e:
        logger.error(f"Error in API call: {str(e)}")
        return jsonify({'error': f"Failed to process question: {str(e)}"}), 500

if __name__ == '__main__':
    """Run the Flask application."""
    app.run(host='0.0.0.0', port=5000)