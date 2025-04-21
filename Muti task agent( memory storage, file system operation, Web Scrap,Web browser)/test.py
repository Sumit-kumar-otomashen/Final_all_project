import os
import json
import shutil
import uuid
from typing import List, Dict, Any, Annotated
from flask import Flask, request, jsonify, render_template
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import TavilySearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pdfplumber
import re
import requests
from bs4 import BeautifulSoup

# Set USER_AGENT environment variable at the top to avoid warnings
os.environ["USER_AGENT"] = "MyLangChainAgent/1.0"

# Initialize Flask app
app = Flask(__name__)

# Global variable to store allowed directories (user configurable)
ALLOWED_DIRECTORIES = []

os.environ["TAVILY_API_KEY"] = "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"

# Utility Functions
def is_allowed_path(path: str) -> bool:
    if not ALLOWED_DIRECTORIES:
        return False
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(os.path.abspath(allowed_dir)) for allowed_dir in ALLOWED_DIRECTORIES)

def normalize_path(path: str) -> str:
    """Convert backslashes to forward slashes and normalize path"""
    return os.path.normpath(path.replace('\\', '/'))

# File System Tools
@tool
def create_directory(path: str, config: RunnableConfig) -> str:
    """
    Creates a directory at the specified path if it is within the allowed directories.
    """
    normalized_path = normalize_path(path)
    if not is_allowed_path(normalized_path):
        return f"Error: Path '{normalized_path}' not in allowed directories."
    try:
        os.makedirs(normalized_path, exist_ok=True)
        return f"Directory '{normalized_path}' created or exists."
    except Exception as e:
        return f"Error creating directory: {str(e)}"

@tool
def list_directory(path: str, config: RunnableConfig) -> str:
    """
    Lists all PDF files in the specified directory if it is within the allowed directories.
    """
    normalized_path = normalize_path(path)
    if not is_allowed_path(normalized_path):
        return f"Error: Path '{normalized_path}' not in allowed directories."
    if not os.path.exists(normalized_path):
        return f"Error: Path '{normalized_path}' does not exist."
    try:
        items = os.listdir(normalized_path)
        pdf_files = [item for item in items if item.lower().endswith('.pdf')]
        return "\n".join(pdf_files) or "No PDF files found."
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@tool
def move_file(source: str, destination: str, config: RunnableConfig) -> str:
    """
    Moves a file from the source path to the destination path if both are within allowed directories.
    """
    normalized_source = normalize_path(source)
    normalized_destination = normalize_path(destination)
    if not (is_allowed_path(normalized_source) and is_allowed_path(normalized_destination)):
        return "Error: Paths not in allowed directories."
    if not os.path.exists(normalized_source):
        return f"Error: Source '{normalized_source}' does not exist."
    try:
        shutil.move(normalized_source, normalized_destination)
        return f"Moved '{normalized_source}' to '{normalized_destination}'."
    except Exception as e:
        return f"Error moving file: {str(e)}"

@tool
def extract_pdf_text(path: str, config: RunnableConfig) -> str:
    """
    Extracts text from a PDF file at the specified path if it is within allowed directories.
    """
    normalized_path = normalize_path(path)
    if not is_allowed_path(normalized_path):
        return f"Error: Path '{normalized_path}' not in allowed directories."
    if not os.path.exists(normalized_path):
        return f"Error: File '{normalized_path}' does not exist."
    try:
        with pdfplumber.open(normalized_path) as pdf:
            text = "".join(page.extract_text() or "" for page in pdf.pages)
        return text or "No text extracted."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

@tool
def directory_tree(path: str, config: RunnableConfig) -> str:
    """
    Generates a JSON representation of the directory tree starting from the specified path if it is within allowed directories.
    """
    normalized_path = normalize_path(path)
    if not is_allowed_path(normalized_path):
        return f"Error: Path '{normalized_path}' not in allowed directories."
    if not os.path.exists(normalized_path):
        return f"Error: Path '{normalized_path}' does not exist."
    try:
        def build_tree(current_path):
            if os.path.isfile(current_path):
                return {"name": os.path.basename(current_path), "type": "file"}
            return {
                "name": os.path.basename(current_path),
                "type": "directory",
                "children": [build_tree(os.path.join(current_path, child)) for child in os.listdir(current_path)]
            }
        tree = build_tree(normalized_path)
        return json.dumps(tree, indent=2)
    except Exception as e:
        return f"Error getting directory tree: {str(e)}"

# Add Directory Tool
@tool
def add_allowed_directory(path: str, config: RunnableConfig) -> str:
    """
    Adds a directory to the list of allowed directories.
    """
    global ALLOWED_DIRECTORIES
    normalized_path = normalize_path(path)
    try:
        abs_path = os.path.abspath(normalized_path)
        if abs_path in ALLOWED_DIRECTORIES:
            return f"Directory '{abs_path}' is already allowed."
        os.makedirs(abs_path, exist_ok=True)
        ALLOWED_DIRECTORIES.append(abs_path)
        return f"Directory '{abs_path}' added to allowed directories. Current allowed directories: {', '.join(ALLOWED_DIRECTORIES)}"
    except Exception as e:
        return f"Error adding directory: {str(e)}"

@tool
def list_allowed_directories(_, config: RunnableConfig) -> str:
    """
    Lists all currently allowed directories.
    """
    if not ALLOWED_DIRECTORIES:
        return "No directories are currently allowed. Please add directories using add_allowed_directory."
    return f"Current allowed directories: {', '.join(ALLOWED_DIRECTORIES)}"

# Web Tools
@tool
def scrape_webpages(urls: List[str], config: RunnableConfig) -> str:
    """
    Scrapes content from the provided list of URLs.
    """
    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        return "\n\n".join(
            [f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>' for doc in docs]
        )
    except Exception as e:
        return f"Error scraping web pages: {str(e)}"

@tool
def scrape_imdb_top_movies(url: str, config: RunnableConfig) -> str:
    """
    Scrapes top movies from IMDb top chart.
    """
    try:
        # Set a user agent to avoid being blocked
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find movie elements - adjust selectors based on IMDb's current structure
        movies = []
        
        # IMDb has different layouts, try common patterns
        # First try to find the new design elements
        movie_elements = soup.select("li.ipc-metadata-list-summary-item")
        
        if movie_elements:
            for i, movie in enumerate(movie_elements[:5]):  # Get top 5 movies
                title_element = movie.select_one(".ipc-title-link-wrapper")
                if title_element:
                    title = title_element.get_text(strip=True)
                    movies.append(f"{i+1}. {title}")
        else:
            # Try older design
            movie_elements = soup.select(".lister-list tr")
            for i, movie in enumerate(movie_elements[:5]):
                title_element = movie.select_one(".titleColumn a")
                if title_element:
                    title = title_element.get_text(strip=True)
                    movies.append(f"{i+1}. {title}")
        
        if not movies:
            return "Failed to extract movies from the page. The website structure might have changed."
        
        return "Top movies on IMDb:\n" + "\n".join(movies)
    except requests.RequestException as e:
        return f"Error accessing IMDb: {str(e)}"
    except Exception as e:
        return f"Error scraping IMDb: {str(e)}"

# Check and set up Tavily API key
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    print("Warning: TAVILY_API_KEY environment variable is not set properly. Some search functionality may be limited.")

tavily_tool = TavilySearchResults(
    max_results=5,
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    tavily_api_key=tavily_api_key
)

# Memory Tools
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_faiss_vectorstore():
    if os.path.exists("faiss_index") and os.path.isdir("faiss_index"):
        try:
            return FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
    starter_doc = Document(
        page_content="Initialization document",
        metadata={"user_id": "system", "id": str(uuid.uuid4())}
    )
    vectorstore = FAISS.from_documents([starter_doc], embedding)
    vectorstore.save_local("faiss_index")
    return vectorstore

recall_vector_store = get_faiss_vectorstore()

def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        return "default_user"
    return user_id

@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """
    Saves a memory string to the recall vector store for the user.
    """
    user_id = get_user_id(config)
    document = Document(
        page_content=memory,
        metadata={"user_id": user_id, "id": str(uuid.uuid4())}
    )
    recall_vector_store.add_documents([document])
    recall_vector_store.save_local("faiss_index")
    return "Memory saved: " + memory[:50] + "..."

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """
    Searches the recall vector store for memories similar to the query.
    """
    user_id = get_user_id(config)
    try:
        documents = recall_vector_store.similarity_search(
            query, k=3, filter={"user_id": user_id}
        )
        return [doc.page_content for doc in documents] if documents else []
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

# Tool List
tools = [
    create_directory, list_directory, move_file, extract_pdf_text, directory_tree,
    scrape_webpages, scrape_imdb_top_movies, tavily_tool, save_recall_memory, search_recall_memories,
    add_allowed_directory, list_allowed_directories
]

# Language Model Setup
try:
    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key="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"
    )
except Exception as e:
    print(f"Warning: Error initializing Groq LLM: {e}")
    # Fallback message for LLM errors
    class FallbackLLM:
        def bind_tools(self, tools):
            return self
        
        def invoke(self, input_data):
            return AIMessage(content="I'm currently having trouble connecting to my knowledge base. Please try again later.")
    
    llm = FallbackLLM()

# System Prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an advanced ReAct agent capable of organizing files, scraping web pages, searching the web, and recalling past information using memory. You reason and act autonomously to complete tasks.

### Task Determination
- **Directory Management**: If the user mentions adding directories or paths, use `add_allowed_directory` to update allowed directories.
- **File Organization**: If the user mentions "organize PDFs" or specifies a directory, follow the file organization workflow.
- **Web Scraping**: If the user provides URLs, use `scrape_webpages`.
- **IMDb Scraping**: If the user wants to scrape IMDb top movies, use `scrape_imdb_top_movies`.
- **Web Searching**: For general queries, check memories first, then use `tavily_search` if needed.
- **Memory Management**: Use `save_recall_memory` to store useful info and `search_recall_memories` internally via recall_memories.

### Directory Configuration
1. Before working with files, check if paths are in allowed directories with `list_allowed_directories`.
2. If needed, add directories with `add_allowed_directory` (must be done before file operations).

### File Organization Workflow
1. Use `list_directory` to list PDFs in the specified directory.
2. For each PDF:
   - Use `extract_pdf_text` to get content.
   - Classify as 'resume' if it has ≥2 keywords from ['experience', 'education', 'skills', 'work history', 'references', 'certifications', 'projects', 'achievements']; else 'other'.
   - Use `create_directory` for 'resume' and 'other' subfolders.
   - Use `move_file` to move the PDF.
3. Use `directory_tree` to show the final structure.

### Guidelines
- Check `recall_memories` before acting.
- Ask user to add directories if needed before file operations.
- Use forward slashes in paths (e.g., '/path/to/directory').
- Save memories as 'User query: [query]\nAssistant answer: [answer]' after tasks.
- Handle errors gracefully.
- Format outputs with markdown.

### Recall Memories
{recall_memories}
"""
        ),
        ("placeholder", "{messages}"),
    ]
)

# State Definition
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    recall_memories: List[str]

# Load Memories Node
def load_memories(state: State, config: RunnableConfig) -> Dict[str, Any]:
    last_user_message = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "")
    recall_memories = search_recall_memories(last_user_message, config)
    return {"recall_memories": recall_memories}

# Assistant Node
class Assistant:
    def __init__(self, runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        try:
            recall_str = "\n\n".join([f"Memory {i+1}:\n{mem}" for i, mem in enumerate(state["recall_memories"])]) if state["recall_memories"] else "No relevant memories found."
            input_data = {
                "recall_memories": recall_str,
                "messages": state["messages"]
            }
            result = self.runnable.invoke(input_data)
            
            if not getattr(result, 'tool_calls', None) and result.content:
                last_user_message = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "")
                memory = f"User query: {last_user_message}\nAssistant answer: {result.content}"
                if len(memory) > 50:
                    save_recall_memory(memory, config)
            
            return {"messages": state["messages"] + [result]}
        except Exception as e:
            error_message = f"I encountered an error processing your request: {str(e)}"
            return {"messages": state["messages"] + [AIMessage(content=error_message)]}

# Parse directory paths from user message
def extract_directory_paths(message: str) -> List[str]:
    pattern = r'(?:[a-zA-Z]:\\[^<>:"/\\|?*\n]+|/[^<>:"/\\|?*\n]+)'
    paths = re.findall(pattern, message)
    return [p.replace('\\', '/') for p in paths]

# State Graph Setup
builder = StateGraph(State)
builder.add_node("load_memories", load_memories)
builder.add_node("assistant", Assistant(primary_assistant_prompt | llm.bind_tools(tools)))
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        user_message = request.json.get('message', '')
        user_id = request.json.get('user_id', 'default_user')
        thread_id = request.json.get('thread_id', str(uuid.uuid4()))
        
        potential_paths = extract_directory_paths(user_message)
        directory_operations = []
        
        for path in potential_paths:
            if "add directory" in user_message.lower() or "allow directory" in user_message.lower():
                # Use invoke instead of __call__ to fix deprecation warning
                config = {"configurable": {"user_id": user_id}}
                result = add_allowed_directory.invoke(path, config)
                directory_operations.append(result)
        
        config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
        state = {"messages": [HumanMessage(content=user_message)], "recall_memories": []}
        
        # Handle IMDb scraping request specially
        if "https://www.imdb.com" in user_message and "scrap" in user_message.lower():
            url_pattern = r'https://www\.imdb\.com\S+'
            imdb_urls = re.findall(url_pattern, user_message)
            if imdb_urls:
                # Use invoke instead of __call__ to fix deprecation warning
                config = {"configurable": {"user_id": user_id}}
                result = scrape_imdb_top_movies.invoke(imdb_urls[0], config)
                return jsonify({
                    "response": result,
                    "thread_id": thread_id,
                    "tools_used": ["scrape_imdb_top_movies"]
                })
        
        result = graph.invoke(state, config)
        
        messages = result["messages"]
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        tool_messages = []
        
        for msg in messages:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_messages.append(f"Used tool: {tool_call['name']}")
        
        final_response = ai_messages[-1].content if ai_messages else "No response generated."
        if directory_operations:
            final_response = "Directory Operations:\n" + "\n".join(directory_operations) + "\n\n" + final_response
        
        return jsonify({
            "response": final_response,
            "thread_id": thread_id,
            "tools_used": tool_messages
        })
    except Exception as e:
        return jsonify({
            "response": f"Error communicating with the server: {str(e)}. Please try again.",
            "thread_id": request.json.get('thread_id', str(uuid.uuid4())),
            "tools_used": []
        }), 500

# HTML Template
def index_template():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangChain File Agent Chatbot</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            #chat-container {
                height: 500px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                margin-bottom: 10px;
                background-color: white;
                border-radius: 5px;
            }
            .message {
                margin-bottom: 10px;
                padding: 8px 12px;
                border-radius: 5px;
                max-width: 80%;
            }
            .user-message {
                background-color: #e1f5fe;
                align-self: flex-end;
                margin-left: auto;
            }
            .bot-message {
                background-color: #f1f1f1;
            }
            #user-input {
                width: calc(100% - 80px);
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            button {
                padding: 10px 15px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            .message-container {
                display: flex;
                flex-direction: column;
            }
            .tool-message {
                font-size: 0.8em;
                color: #666;
                margin-top: 5px;
                font-style: italic;
            }
            .loading {
                display: none;
                margin: 10px 0;
                font-style: italic;
                color: #666;
            }
            .help-text {
                font-size: 0.9em;
                color: #666;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <h1>LangChain File Agent Chatbot</h1>
        
        <div class="help-text">
            <p>This agent can help you organize files, scrape web pages, and more. Start by adding an allowed directory:</p>
            <code>Add directory C:/path/to/your/folder</code>
        </div>
        
        <div id="chat-container"></div>
        
        <div class="loading" id="loading-indicator">Thinking...</div>
        
        <div style="display: flex; margin-top: 10px;">
            <input type="text" id="user-input" placeholder="Type your message here...">
            <button onclick="sendMessage()">Send</button>
        </div>

        <script>
            const userId = 'user_' + Math.random().toString(36).substring(2, 10);
            let threadId = null;
            
            function addMessage(message, sender) {
                const chatContainer = document.getElementById('chat-container');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message-container';
                
                const messageContent = document.createElement('div');
                messageContent.className = 'message ' + (sender === 'user' ? 'user-message' : 'bot-message');
                
                if (sender === 'bot') {
                    const converter = new showdown.Converter();
                    messageContent.innerHTML = converter.makeHtml(message);
                } else {
                    messageContent.textContent = message;
                }
                
                messageDiv.appendChild(messageContent);
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function addToolMessage(toolUsage) {
                const chatContainer = document.getElementById('chat-container');
                const lastMessageContainer = chatContainer.lastChild;
                
                const toolDiv = document.createElement('div');
                toolDiv.className = 'tool-message';
                toolDiv.textContent = toolUsage;
                
                lastMessageContainer.appendChild(toolDiv);
            }
            
            function sendMessage() {
                const userInput = document.getElementById('user-input');
                const message = userInput.value.trim();
                
                if (message) {
                    addMessage(message, 'user');
                    userInput.value = '';
                    document.getElementById('loading-indicator').style.display = 'block';
                    
                    fetch('/send_message', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            message: message,
                            user_id: userId,
                            thread_id: threadId
                        }),
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Network response was not ok');
                        }
                        return response.json();
                    })
                    .then(data => {
                        document.getElementById('loading-indicator').style.display = 'none';
                        addMessage(data.response, 'bot');
                        threadId = data.thread_id;
                        if (data.tools_used && data.tools_used.length > 0) {
                            data.tools_used.forEach(tool => addToolMessage(tool));
                        }
                    })
                    .catch(error => {
                        document.getElementById('loading-indicator').style.display = 'none';
                        console.error('Error:', error);
                        addMessage('Error communicating with the server. Please try again.', 'bot');
                    });
                }
            }
            
            document.getElementById('user-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
            
            // Error handling for script loading
            window.addEventListener('error', function(e) {
                console.error('Script error:', e);
                if (e.target && e.target.src && e.target.src.includes('showdown.min.js')) {
                    alert('Could not load required libraries. The chat may not function correctly.');
                }
            }, true);
            
            window.onload = function() {
                addMessage('👋 Hello! I am your file organization assistant. Before we start working with files, please add an allowed directory using "Add directory /path/to/your/folder"', 'bot');
            }
        </script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/showdown/2.1.0/showdown.min.js"></script>
    </body>
    </html>
    """

# Setup Application
def setup_app():
    os.makedirs('templates', exist_ok=True)
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(index_template())

if __name__ == "__main__":
    setup_app()
    app.run(debug=True, port=5000)