import os
import json
import shutil
import uuid
import logging
from typing import List, Dict, Any, Annotated, Optional
from flask import Flask, request, jsonify, render_template
from langchain_core.tools import StructuredTool
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
from pydantic import BaseModel, Field

# Set USER_AGENT to avoid warnings
os.environ["USER_AGENT"] = "MyLangChainAgent/1.0"

# Initialize Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Global variable for allowed directories
ALLOWED_DIRECTORIES = []

# Set environment variable for Tavily API
os.environ["TAVILY_API_KEY"] = "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"

# Utility Functions
def is_allowed_path(path: str) -> bool:
    """Check if a path is within allowed directories."""
    if not ALLOWED_DIRECTORIES:
        return False
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(os.path.abspath(allowed_dir)) for allowed_dir in ALLOWED_DIRECTORIES)

def normalize_path(path: str) -> str:
    """Convert backslashes to forward slashes and normalize path."""
    return os.path.normpath(path.replace('\\', '/'))

# File System Tools
def create_directory(path: str, config: RunnableConfig) -> str:
    """Creates a directory if within allowed directories."""
    normalized_path = normalize_path(path)
    if not is_allowed_path(normalized_path):
        return f"Error: Path '{normalized_path}' not in allowed directories."
    try:
        os.makedirs(normalized_path, exist_ok=True)
        return f"Directory '{normalized_path}' created or exists."
    except Exception as e:
        return f"Error creating directory: {str(e)}"

def list_directory(path: str, config: RunnableConfig) -> str:
    """Lists PDF files in a directory if allowed."""
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

def move_file(source: str, destination: str, config: RunnableConfig) -> str:
    """Moves a file if both paths are allowed."""
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

def extract_pdf_text(path: str, config: RunnableConfig) -> str:
    """Extracts text from a PDF if path is allowed."""
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

def directory_tree(path: str, config: RunnableConfig) -> str:
    """Generates a JSON directory tree if path is allowed."""
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

# Directory Management Tools
def add_allowed_directory(path: str, config: RunnableConfig) -> str:
    """Adds a directory to allowed list."""
    global ALLOWED_DIRECTORIES
    normalized_path = normalize_path(path)
    try:
        abs_path = os.path.abspath(normalized_path)
        if abs_path in ALLOWED_DIRECTORIES:
            return f"Directory '{abs_path}' is already allowed."
        os.makedirs(abs_path, exist_ok=True)
        ALLOWED_DIRECTORIES.append(abs_path)
        return f"Directory '{abs_path}' added. Allowed directories: {', '.join(ALLOWED_DIRECTORIES)}"
    except Exception as e:
        return f"Error adding directory: {str(e)}"

def list_allowed_directories(_: str, config: RunnableConfig) -> str:
    """Lists all allowed directories."""
    return f"Allowed directories: {', '.join(ALLOWED_DIRECTORIES)}" if ALLOWED_DIRECTORIES else "No allowed directories."

# Web Tools
def scrape_webpages(urls: List[str], config: RunnableConfig) -> str:
    """Scrapes content from URLs."""
    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        return "\n\n".join(
            [f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>' for doc in docs]
        )
    except Exception as e:
        return f"Error scraping web pages: {str(e)}"

# Tavily Search Tool
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not set. Set it via environment or .env file.")

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
    """Initialize or load FAISS vector store."""
    if os.path.exists("faiss_index") and os.path.isdir("faiss_index"):
        try:
            return FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
    starter_doc = Document(
        page_content="Initialization document",
        metadata={"user_id": "system", "id": str(uuid.uuid4())}
    )
    vectorstore = FAISS.from_documents([starter_doc], embedding)
    vectorstore.save_local("faiss_index")
    return vectorstore

recall_vector_store = get_faiss_vectorstore()

def get_user_id(config: dict) -> str:
    """Get user ID from config."""
    user_id = config.get("configurable", {}).get("user_id")
    if user_id is None:
        raise ValueError("User ID required for memory operations.")
    return user_id

def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Saves a memory to the vector store."""
    user_id = get_user_id(config)
    document = Document(
        page_content=memory,
        metadata={"user_id": user_id, "id": str(uuid.uuid4())}
    )
    recall_vector_store.add_documents([document])
    recall_vector_store.save_local("faiss_index")
    return "Memory saved: " + memory[:50] + "..."

# Custom Tool for search_recall_memories
class SearchRecallMemoriesInput(BaseModel):
    query: str = Field(description="The search query")
    user_id: str = Field(description="The user ID")

def search_recall_memories_func(query: str, user_id: str) -> List[str]:
    """Search recall memories for a given query."""
    try:
        documents = recall_vector_store.similarity_search(
            query, k=3, filter={"user_id": user_id}
        )
        return [doc.page_content for doc in documents] if documents else []
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []

search_recall_memories = StructuredTool.from_function(
    func=search_recall_memories_func,
    name="search_recall_memories",
    description="Search recall memories for a given query.",
    args_schema=SearchRecallMemoriesInput
)

# Tool List
tools = [
    StructuredTool.from_function(create_directory, name="create_directory", description="Creates a directory"),
    StructuredTool.from_function(list_directory, name="list_directory", description="Lists PDF files in a directory"),
    StructuredTool.from_function(move_file, name="move_file", description="Moves a file"),
    StructuredTool.from_function(extract_pdf_text, name="extract_pdf_text", description="Extracts text from a PDF"),
    StructuredTool.from_function(directory_tree, name="directory_tree", description="Generates a JSON directory tree"),
    StructuredTool.from_function(scrape_webpages, name="scrape_webpages", description="Scrapes content from URLs"),
    tavily_tool,
    StructuredTool.from_function(save_recall_memory, name="save_recall_memory", description="Saves a memory"),
    search_recall_memories,
    StructuredTool.from_function(add_allowed_directory, name="add_allowed_directory", description="Adds a directory to allowed list"),
    StructuredTool.from_function(list_allowed_directories, name="list_allowed_directories", description="Lists all allowed directories")
]

# Language Model Setup
llm = ChatGroq(
    model="qwen-qwq-32b",
    api_key="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"
)

# System Prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an advanced ReAct agent for organizing files, scraping web pages, searching the web, and recalling memories.

### Task Determination
- **Directory Management**: Use `add_allowed_directory` for directory-related requests.
- **File Organization**: For "organize PDFs" or directory mentions, follow the file workflow.
- **Web Scraping**: Use `scrape_webpages` for URLs.
- **Web Searching**: Check memories, then use `tavily_search` if needed.
- **Memory**: Use `save_recall_memory` and `search_recall_memories`.

### Directory Configuration
1. Check paths with `list_allowed_directories` before file ops.
2. Add directories with `add_allowed_directory` if needed.

### File Organization Workflow
1. List PDFs with `list_directory`.
2. For each PDF:
   - Extract text with `extract_pdf_text`.
   - Classify as 'resume' (≥2 keywords: 'experience', 'education', 'skills', etc.) or 'other'.
   - Create subfolders with `create_directory`.
   - Move with `move_file`.
3. Show structure with `directory_tree`.

### Guidelines
- Check `recall_memories` first.
- Request directories if needed for files.
- Use forward slashes in paths.
- Save memories as 'User query: [query]\nAssistant answer: [answer]'.
- Handle errors gracefully.
- Use markdown for outputs.

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

# Nodes
def load_memories(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Load relevant memories using the tool's invoke method."""
    last_user_message = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "")
    user_id = get_user_id(config)
    input_data = {"query": last_user_message, "user_id": user_id}
    recall_memories = search_recall_memories.invoke(input_data, config)
    return {"recall_memories": recall_memories}

class Assistant:
    def __init__(self, runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        recall_str = "\n\n".join([f"Memory {i+1}:\n{mem}" for i, mem in enumerate(state["recall_memories"])]) if state["recall_memories"] else "No relevant memories."
        input_data = {"recall_memories": recall_str, "messages": state["messages"]}
        result = self.runnable.invoke(input_data)
        
        if isinstance(result, dict):
            error_msg = result.get('error', 'Unknown error occurred.')
            return {"messages": state["messages"] + [AIMessage(content=f"Error: {error_msg}")]}

        if not result.tool_calls and result.content:
            last_user_message = next((msg.content for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), "")
            memory = f"User query: {last_user_message}\nAssistant answer: {result.content}"
            if len(memory) > 50:
                save_recall_memory(memory, config)
        
        return {"messages": state["messages"] + [result]}

# Parse directory paths
def extract_directory_paths(message: str) -> List[str]:
    pattern = r'(?:[a-zA-Z]:\\[^<>:"/\\|?*\n]+|/[^<>:"/\\|?*\n]+)'
    paths = re.findall(pattern, message)
    return [p.replace('\\', '/') for p in paths]

# State Graph
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
    """Handle user messages with error handling."""
    try:
        user_message = request.json.get('message', '')
        user_id = request.json.get('user_id', 'default_user')
        thread_id = request.json.get('thread_id', str(uuid.uuid4()))
        
        potential_paths = extract_directory_paths(user_message)
        directory_operations = []
        
        for path in potential_paths:
            if "add directory" in user_message.lower() or "allow directory" in user_message.lower():
                result = add_allowed_directory(path, {"configurable": {"user_id": user_id}})
                directory_operations.append(result)
        
        config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
        state = {"messages": [HumanMessage(content=user_message)]}
        
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
        
        response_data = {
            "response": final_response,
            "thread_id": thread_id,
            "tools_used": tool_messages
        }
    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}", exc_info=True)
        response_data = {
            "response": f"Error: {str(e)}",
            "thread_id": thread_id,
            "tools_used": []
        }
    
    return jsonify(response_data)

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
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
            #chat-container { height: 500px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; background-color: white; border-radius: 5px; }
            .message { margin-bottom: 10px; padding: 8px 12px; border-radius: 5px; max-width: 80%; }
            .user-message { background-color: #e1f5fe; align-self: flex-end; margin-left: auto; }
            .bot-message { background-color: #f1f1f1; }
            #user-input { width: calc(100% - 80px); padding: 10px; border: 1px solid #ccc; border-radius: 5px; }
            button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; border-radius: 5px; cursor: pointer; }
            .message-container { display: flex; flex-direction: column; }
            .tool-message { font-size: 0.8em; color: #666; margin-top: 5px; font-style: italic; }
            .loading { display: none; margin: 10px 0; font-style: italic; color: #666; }
            .help-text { font-size: 0.9em; color: #666; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1>LangChain File Agent Chatbot</h1>
        <div class="help-text">
            <p>This agent can organize files, scrape web pages, and more. Start by adding a directory:</p>
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
                    .then(response => response.json())
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
                        addMessage('Error: An error occurred.', 'bot');
                    });
                }
            }
            
            document.getElementById('user-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
            
            window.onload = function() {
                addMessage('👋 Hello! I am your file organization assistant. Add a directory with "Add directory /path/to/your/folder" to start.', 'bot');
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
    app.run(host='0.0.0.0', port=5000)  # Runs on all interfaces, port 5000, no debug mode