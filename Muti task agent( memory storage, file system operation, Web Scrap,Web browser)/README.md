# LangGraph Chatbot

A Flask-based web application that uses LangGraph to create an intelligent assistant capable of organizing files, searching the web, and maintaining conversation memory.

## Features

- **Interactive Chat Interface**: Modern web UI for conversing with the assistant
- **File Organization**: Organize PDFs into categories based on content analysis
- **Web Search**: Search the web for information using Tavily Search API
- **Web Scraping**: Extract content from web pages
- **Memory System**: Remember past conversations and use them for context
- **PDF Processing**: Extract text from PDF files

## Tech Stack

- **Backend**: Flask, LangGraph
- **LLM**: Groq (llama3-70b-8192)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for memory storage
- **Frontend**: HTML, CSS, JavaScript

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd langgraph-chatbot
   ```

2. Run the setup script:
   ```
   python setup.py
   ```
   This will:
   - Create necessary directories
   - Install required packages
   - Set up the templates directory

3. Set up environment variables:
   ```
   export LANGSMITH_TRACING=true
   export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
   export LANGSMITH_API_KEY=your_langsmith_api_key
   export LANGSMITH_PROJECT=langgraph
   export TAVILY_API_KEY=your_tavily_api_key
   export USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

- **app.py**: Flask application server
- **agent.py**: LangGraph agent implementation
- **setup.py**: Script to set up the project
- **templates/index.html**: Chat interface
- **faiss_index/**: Directory for storing vector embeddings

## Agent Capabilities

The LangGraph agent can:

1. **Organize Files**:
   - List PDFs in specified directories
   - Extract text from PDFs
   - Categorize PDFs based on content analysis
   - Create directories for different categories
   - Move files to appropriate directories

2. **Web Interactions**:
   - Search the web using Tavily
   - Scrape content from web pages

3. **Memory Management**:
   - Store important conversation details
   - Retrieve relevant memories for context

## Usage Examples

### File Organization
```
Please organize my PDF files in C:\Users\Sumit\Desktop\testing\Test data file\pdf
```

### Web Search
```
What are the latest developments in AI?
```

### Memory Recall
```
What did we talk about yesterday?
```

## Configuration

The agent is configured to use only specified directories for file operations. By default, the allowed directory is:
```
C:\Users\Sumit\Desktop\testing\Test data file\pdf
```

To modify this, edit the `ALLOWED_DIRECTORIES` list in `agent.py`.

## License

[Your License Here]

## Acknowledgements

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [Groq](https://groq.com/) and [LangChain](https://langchain.com/)