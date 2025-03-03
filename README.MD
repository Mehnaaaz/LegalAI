# LegalMind - AI Legal Document Analysis

LegalMind is a web application that enables users to upload legal documents (PDF, DOCX, TXT) and ask questions about the content. It uses AI to analyze the documents and provide answers based on the context of the uploaded files.

## Technologies Used

- **Backend**: Flask, Flask-CORS
- **Frontend**: HTML, CSS, JavaScript
- **AI and NLP**: Sentence Transformers, Together AI API
- **Other Libraries**: pdfplumber, docx2txt, numpy, python-dotenv

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/LegalMind.git
   cd LegalMind
   ```

2. **Install Dependencies**

   Make sure you have Python 3.8 or higher installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**

   Create a `.env` file in the root directory and add your Together AI API key:

   ```bash
   TOGETHER_API_KEY=your_api_key_here
   ```

4. **Run the Application**

   Start the Flask server:

   ```bash
   python app.py
   ```

   The application will be available at `http://localhost:5000`.

## Usage

1. **Upload a Document**

   - Drag and drop a legal document (PDF, DOCX, or TXT) into the upload area or click to select a file.
   - The document will be processed, and you'll see a confirmation message.

2. **Ask Questions**

   - Type your question in the input field at the bottom of the chat box.
   - Press Enter or click the send button (paper plane icon) to submit your question.
   - The AI will analyze the document and provide an answer based on the content.

3. **View Responses**

   - Responses will appear in the chat box, along with the questions you asked.
   - If the AI is processing your question, you'll see a "Analyzing document and generating response..." message.

## Limitations

- The application is designed for legal documents but can be used with any text-based documents.
- Very large documents may take longer to process and could potentially cause performance issues.
- The accuracy of the answers depends on the quality of the document and the specificity of the questions.
- Currently, only PDF, DOCX, and TXT files are supported.

## Dependencies

- **Flask**: Web framework for the backend.
- **Flask-CORS**: Handles Cross-Origin Resource Sharing.
- **python-dotenv**: Loads environment variables from a `.env` file.
- **sentence-transformers**: Generates embeddings for semantic search.
- **pdfplumber**: Extracts text from PDF files.
- **docx2txt**: Extracts text from DOCX files.
- **numpy**: Handles numerical operations, used for similarity calculations.
- **together**: Client for the Together AI API.
- **huggingface_hub**: Access to Hugging Face models (used by sentence-transformers).
- **torch**: PyTorch, required by sentence-transformers.
- **gunicorn**: WSGI server for deploying the application.

## Project Structure

- `app.py`: The main Flask application script.
- `templates/index.html`: The frontend HTML file.
- `.env`: Environment variables (not tracked in Git).
- `requirements.txt`: List of Python dependencies.

## Contributing

If you find any issues or have suggestions for improvements, please open an issue on the GitHub repository. Contributions are welcome via pull requests.