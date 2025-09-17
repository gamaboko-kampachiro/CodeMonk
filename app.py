"""
Universal Document Intelligence Chatbot
======================================

An intelligent chatbot that answers queries using both uploaded documents and web search,
depending on the nature of the question.

Features:
- Universal Document Processing (PDF support)
- Smart Query Routing (document vs web search)
- Semantic Vector Search with FAISS
- Interactive Chat Interface with Gradio
"""

import os
import json
import logging
from typing import List, Tuple, Dict
import gradio as gr
import ollama
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotApp:
    def __init__(self):
        """Initialize the chatbot application"""
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        self.chat_history: List[Dict[str, str]] = []
        self.uploaded_docs = []
        
        # Web search settings
        self.serper_api_key = "5a5100045641d233a3194d3ba265946f3a964ac6"
        
        # Query routing keywords
        self.web_search_keywords = [
            "latest", "current", "2024", "2025", "today", "recent",
            "explain", "how does", "how do", "what is", "what are",
            "vs", "alternatives to", "compared to", "comparison",
            "trends", "price", "stock", "market", "news",
            "specifications", "specs", "features",
            "difference", "different"
        ]
        
        # Check available Ollama models
        self.available_models = self._get_ollama_models()
        self.current_model = self._select_model()
        
    def _get_ollama_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = ollama.list()
            models = [model['name'] for model in response['models']]
            logger.info(f"Available Ollama models: {models}")
            return models
        except Exception as e:
            logger.error(f"Error getting Ollama models: {e}")
            return ["llama3.2"]  # Default fallback
    
    def _select_model(self) -> str:
        """Select the best available model"""
        preferred_models = ["gemma2:2b", "llama3.2", "phi3", "llama3", "mistral"]
        
        # If we have available models, try to match with preferred ones
        if self.available_models:
            for model in preferred_models:
                # Check if model exists in available models (with or without version)
                for available_model in self.available_models:
                    if model.split(':')[0] in available_model:
                        logger.info(f"Selected model: {available_model}")
                        return available_model
            
            # If none of the preferred models are available, use the first available
            selected = self.available_models[0]
            logger.info(f"Fallback to first available model: {selected}")
            return selected
        
        # Default fallback if no models are available
        logger.warning("No models available, using default phi3")
        return "phi3"
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            doc.close()
            logger.info(f"Extracted text from {pdf_path}, {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def process_file(self, file) -> List[str]:
        """Process a single file and return text chunks"""
        try:
            file_extension = os.path.splitext(file.name)[1].lower()
            
            if file_extension == ".pdf":
                text = self.extract_text_from_pdf(file.name)
            else:
                logger.warning(f"Unsupported file format: {file_extension}")
                return []
            
            if text:
                # Split text into chunks
                chunks = self.text_splitter.split_text(text)
                logger.info(f"Processed {file.name} into {len(chunks)} chunks")
                return chunks
            else:
                logger.warning(f"No text extracted from {file.name}")
                return []
                
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            return []
    
    def process_files(self, files) -> List[str]:
        """Process multiple files and return all text chunks"""
        all_chunks = []
        
        for file in files:
            chunks = self.process_file(file)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(files)} files into {len(all_chunks)} chunks total")
        return all_chunks
    
    def process_documents(self, files) -> str:
        """Process uploaded documents and create vector store"""
        try:
            if not files:
                return "No files uploaded."
            
            # Process documents
            processed_texts = self.process_files(files)
            
            if processed_texts:
                # Create vector store
                self.vector_store = FAISS.from_texts(processed_texts, self.embeddings)
                self.uploaded_docs = [os.path.basename(file.name) for file in files]
                return f"Successfully processed {len(files)} document(s): {', '.join(self.uploaded_docs)}"
            else:
                return "No text could be extracted from the uploaded files."
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return f"Error processing documents: {str(e)}"
    
    def web_search(self, query: str) -> str:
        """Perform web search and return formatted results"""
        try:
            payload = json.dumps({
                "q": query,
                "num": 5  # Number of results to return
            })
            
            headers = {
                'X-API-KEY': self.serper_api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post("https://google.serper.dev/search", headers=headers, data=payload)
            response.raise_for_status()
            
            results = response.json()
            
            # Format the results
            formatted_results = self._format_results(results)
            
            logger.info(f"Web search for '{query}' returned {len(formatted_results)} characters")
            return formatted_results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Web search request failed: {e}")
            return f"Error performing web search: {str(e)}"
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return f"Error processing web search results: {str(e)}"
    
    def _format_results(self, results: dict) -> str:
        """Format search results into a readable string"""
        try:
            if 'organic' not in results:
                return "No relevant web search results found."
            
            formatted = []
            for i, item in enumerate(results['organic'][:3]):  # Top 3 results
                title = item.get('title', 'No title')
                snippet = item.get('snippet', 'No description')
                link = item.get('link', 'No link')
                
                formatted.append(f"{i+1}. {title}\n   {snippet}\n   Source: {link}\n")
            
            return "\n".join(formatted) if formatted else "No relevant results found."
            
        except Exception as e:
            logger.error(f"Error formatting search results: {e}")
            return "Error formatting search results."
    
    def should_use_web_search(self, query: str) -> bool:
        """
        Determine if web search should be used based on query content
        
        Returns:
            bool: True if web search should be used, False for document search
        """
        try:
            query_lower = query.lower()
            
            # Check for web search keywords
            for keyword in self.web_search_keywords:
                if keyword in query_lower:
                    logger.info(f"Routing to web search due to keyword: {keyword}")
                    return True
            
            # If no keywords match, use document search
            logger.info("Routing to document search")
            return False
            
        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            # Default to document search on error
            return False
    
    def chatbot_response(self, query: str, history: List[dict]) -> Tuple[str, List[dict]]:
        """Generate response based on query and available information"""
        try:
            # Add user query to history
            self.chat_history.append({"role": "user", "content": query})
            
            # Determine routing strategy
            use_web_search = self.should_use_web_search(query)
            source = "Web" if use_web_search else "Document"
            
            response = ""
            
            # If we should use web search or don't have documents
            if use_web_search or self.vector_store is None:
                # Perform web search
                search_results = self.web_search(query)
                
                # Generate response using Ollama with web search context
                prompt = f"""Based on the following web search results, answer the question: {query}
                
Web search results:
{search_results}

Please provide a comprehensive answer based on the search results above."""
                
                ollama_response = ollama.generate(model=self.current_model, prompt=prompt)
                response = ollama_response['response']
                
            # If we have documents and shouldn't use web search
            elif self.vector_store is not None and not use_web_search:
                # Perform similarity search
                docs = self.vector_store.similarity_search(query, k=3)
                
                if docs:
                    # Combine document content
                    context = "\n".join([doc.page_content for doc in docs])
                    
                    # Generate response using Ollama with document context
                    prompt = f"""Based on the following documents, answer the question: {query}
                    
Document content:
{context}

Please provide a comprehensive answer based on the document content above."""
                    
                    ollama_response = ollama.generate(model=self.current_model, prompt=prompt)
                    response = ollama_response['response']
                else:
                    # Fallback to web search if no relevant documents found
                    search_results = self.web_search(query)
                    prompt = f"""Based on the following web search results, answer the question: {query}
                    
Web search results:
{search_results}

Please provide a comprehensive answer based on the search results above."""
                    
                    ollama_response = ollama.generate(model=self.current_model, prompt=prompt)
                    response = ollama_response['response']
                    source = "Web (fallback)"
            
            # Add response to history
            self.chat_history.append({"role": "assistant", "content": f"{response}"})
            
            # Update history for Gradio
            updated_history = self.chat_history.copy()
            
            return "", updated_history
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_msg = f"Error generating response: {str(e)}"
            self.chat_history.append({"role": "assistant", "content": error_msg})
            return "", self.chat_history.copy()
    
    def clear_chat(self) -> Tuple[str, List[dict]]:
        """Clear chat history"""
        self.chat_history = []
        return "", []
    
    def get_system_info(self) -> str:
        """Get system information"""
        info = f"Current Model: {self.current_model}\n"
        info += f"Available Models: {', '.join(self.available_models)}\n"
        info += f"Uploaded Documents: {len(self.uploaded_docs)}\n"
        info += f"Vector Store Status: {'Initialized' if self.vector_store else 'Not initialized'}"
        return info
    
    def get_uploaded_docs_list(self) -> str:
        """Get formatted list of uploaded documents"""
        return ", ".join(self.uploaded_docs)
    
    def process_documents_and_update(self, files) -> str:
        """Process documents and return status message"""
        status = self.process_documents(files)
        # Update system info after processing
        return status
    
    def process_and_update_all(self, files):
        """Process documents and return both status and document list"""
        status_msg = self.process_documents_and_update(files)
        doc_list_msg = self.get_uploaded_docs_list()
        return status_msg, doc_list_msg

# Initialize the chatbot app
chatbot_app = ChatbotApp()

# Create Gradio interface
with gr.Blocks(title="Universal Document Intelligence Chatbot") as demo:
    gr.Markdown("# 🤖 Universal Document Intelligence Chatbot")
    gr.Markdown("Upload documents and ask questions. The chatbot will intelligently use either the documents or web search to answer your queries.")
    
    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(
            label="Conversation",
            type="messages",  # Fix for Gradio 5.x warning
            avatar_images=(
                "https://cdn-icons-png.flaticon.com/512/4712/4712035.png",  # User avatar
                "https://cdn-icons-png.flaticon.com/512/4712/4712139.png"   # Bot avatar
            )
        )
        msg = gr.Textbox(
            label="Your Question",
            placeholder="Ask something about your documents or anything else...",
            container=False
        )
        clear = gr.Button("Clear Chat", variant="secondary")
        
        msg.submit(chatbot_app.chatbot_response, [msg, chatbot], [msg, chatbot])
        clear.click(chatbot_app.clear_chat, None, [msg, chatbot], queue=False)
    
    with gr.Tab("System Info"):
        system_info = gr.Textbox(label="System Information", lines=10, interactive=False)
        refresh_info = gr.Button("Refresh System Info")
        model_selector = gr.Dropdown(
            choices=chatbot_app.available_models,
            value=chatbot_app.current_model,
            label="Select Model",
            allow_custom_value=True  # Add this to prevent warnings
        )
        
        refresh_info.click(chatbot_app.get_system_info, None, system_info)
        # Initialize with system info
        demo.load(chatbot_app.get_system_info, None, system_info)
    
    with gr.Tab("Documents"):
        file_output = gr.File(file_count="multiple", label="Upload PDF Documents")
        upload_btn = gr.Button("Process Documents", variant="primary")
        status = gr.Textbox(label="Status", interactive=False, lines=3)
        doc_list = gr.Textbox(label="Uploaded Documents", interactive=False)
        refresh_docs = gr.Button("Refresh Document List")
        
        # Process documents and update displays
        def process_and_update_all(files):
            """Process documents and return both status and document list"""
            status_msg = chatbot_app.process_documents_and_update(files)
            doc_list_msg = chatbot_app.get_uploaded_docs_list()
            return status_msg, doc_list_msg
        
        def refresh_all_displays():
            """Refresh both system info and document list"""
            return chatbot_app.get_system_info(), chatbot_app.get_uploaded_docs_list()
        
        upload_btn.click(process_and_update_all, file_output, [status, doc_list])
        upload_btn.click(refresh_all_displays, None, [system_info, doc_list])
        refresh_docs.click(chatbot_app.get_uploaded_docs_list, None, doc_list, queue=False)

if __name__ == "__main__":
    demo.launch()
    # Keep the main thread alive
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
