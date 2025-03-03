# LegalMind - AI-Powered Legal Document Analysis

**LegalMind is a sophisticated AI-powered tool designed to assist in analyzing legal documents efficiently.** Simply upload a PDF, DOCX, or TXT file, ask relevant questions, and receive context-aware responses based on the document's content. 

## Key Features
- **Time-Efficient** ⏳: Quickly extract and understand key information from legal documents.
- **AI-Powered Insights** ⚡: Uses advanced natural language processing (NLP) for precise answers.
- **Privacy-Centric** 🔐: Ensures your documents remain securely within your environment.
- **Seamless Integration** 🤖: Combines semantic search with Together AI’s large language models.

## Technologies Used

- **Backend**: Flask, Flask-CORS
- **Frontend**: HTML, CSS, JavaScript
- **AI and NLP**: Sentence Transformers, Together AI API
- **Libraries**: pdfplumber, docx2txt, numpy, python-dotenv

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/LegalMind.git
   cd LegalMind
   ```

2. **Install Dependencies**

   Ensure you have Python 3.8 or higher installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**

   Create a `.env` file in the root directory and add your API key:

   ```bash
   TOGETHER_API_KEY=your_api_key_here
   ```

4. **Run the Application**

   Start the Flask server:

   ```bash
   python app.py
   ```

   The application will be accessible at `http://localhost:5000`.

## How to Use

1. **Upload a Document** 📂
   - Drag and drop a legal document (PDF, DOCX, or TXT) into the upload area or manually select a file.
   - The system processes the document and confirms successful ingestion.

2. **Ask Questions** 💬
   - Enter your legal query in the input field.
   - Press Enter or click the send button to submit.
   - Receive AI-generated responses based on document context.

3. **Review Responses** 🔍
   - View AI-generated answers directly in the chat interface.
   - Ensure responses align with document context for accuracy.

## Limitations
- **Supported Formats**: Currently, only PDF, DOCX, and TXT files are processed.
- **Processing Time**: Large documents may require additional time for analysis.
- **Answer Precision**: Response accuracy depends on document clarity and query specificity.

## Dependencies
- **Flask**: Web framework for backend processing.
- **Flask-CORS**: Manages Cross-Origin Resource Sharing.
- **python-dotenv**: Loads secure environment variables.
- **sentence-transformers**: Converts text into embeddings for semantic search.
- **pdfplumber**: Extracts text from PDFs.
- **docx2txt**: Processes DOCX file content.
- **numpy**: Supports mathematical computations for similarity calculations.
- **together**: Provides access to Together AI models.
- **huggingface_hub**: Connects to Hugging Face NLP resources.
- **torch**: Supports deep learning computations.
- **gunicorn**: WSGI server for deployment.

## Project Structure

- `app.py`: The main Flask application script.
- `templates/index.html`: Frontend UI.
- `.env`: Environment configuration (excluded from version control).
- `requirements.txt`: List of required dependencies.

## Contributing
If you have suggestions or encounter issues, feel free to open an issue or submit a pull request on the GitHub repository. Let’s enhance legal AI together!
