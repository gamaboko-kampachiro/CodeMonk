# Universal Document Intelligence Chatbot

An intelligent chatbot that answers queries using both uploaded documents and web search, depending on the nature of the question.

## 🌟 Features

1. **Universal Document Processing**
   - Support for PDF file formats
   - Intelligent text extraction and chunking
   - Metadata preservation

2. **Smart Query Routing**
   - Document Mode: Answers from uploaded documents
   - Web Search Mode: Uses external search when needed

3. **Semantic Vector Search**
   - Document content embedding using sentence transformers
   - Semantic retrieval with FAISS vector store

4. **Interactive Chat Interface**
   - Upload and manage documents
   - Display chat history
   - Session management

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Ollama with language models
- Internet connection for web search functionality

### Installation

1. Clone or download this repository

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is installed and running on your system:
   - Download from: https://ollama.com/
   - Pull a model: `ollama pull gemma2:2b` (or another supported model)

### Usage

Run the application:
```bash
python app.py
```

The application will start and provide a local URL (typically http://localhost:7860) where you can access the chatbot interface.

## 🎯 How It Works

1. **Upload Documents**: Go to the "Documents" tab and upload PDF files.
2. **Process Documents**: Click "Process Documents" to ingest them into the vector store.
3. **Ask Questions**: Switch to the "Chat" tab and start asking questions.
4. **Intelligent Routing**: The system automatically decides whether to use document content or web search based on your query.

### Query Routing Logic

The system triggers web search when queries include:
- Temporal keywords: latest, 2024, current, etc.
- Explanations: explain, how does, etc.
- Comparisons: vs, alternatives to
- Current data: trends, price, stock
- Specifications not in docs

## 🧩 Components

- **LLM**: Ollama models (gemma2:2b by default)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS
- **Web Search**: Serper.dev API
- **Document Processing**: PyMuPDF
- **Framework**: Langchain
- **UI**: Gradio

## 📁 Project Structure

```
universal-document-chatbot/
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── vector_db/            # Vector database storage directory
├── sample_document.pdf   # Sample document for testing
└── README.md             # This file
```

## 🛠️ Configuration

The application uses the following configuration:

- Default Ollama model: gemma2:2b
- Embedding model: all-MiniLM-L6-v2
- Text chunk size: 1000 characters
- Text overlap: 200 characters
- Serper.dev API key: (provided in code)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.