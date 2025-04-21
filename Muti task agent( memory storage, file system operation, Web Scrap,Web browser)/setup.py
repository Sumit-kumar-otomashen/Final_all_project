import os
import subprocess
import sys

def setup_directories():
    """
    Create necessary directories for the Flask application
    """
    # Create templates directory for HTML files
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created templates directory")
    
    # Create static directory for CSS, JS, and other static files
    if not os.path.exists('static'):
        os.makedirs('static')
        print("Created static directory")
    
    # Create directory for FAISS index
    if not os.path.exists('faiss_index'):
        os.makedirs('faiss_index')
        print("Created faiss_index directory")
    
    # Create directory for PDF files
    pdf_dir = os.path.join('Test data file', 'pdf')
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"Created test PDF directory: {pdf_dir}")

def install_requirements():
    """Install required Python packages"""
    required_packages = [
        "flask",
        "langchain-core",
        "langchain-groq",
        "langgraph",
        "langchain-huggingface",
        "langchain-community",
        "pdfplumber",
        "faiss-cpu",
        "tavily-python"
    ]
    
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + required_packages)
    print("All packages installed successfully")

def copy_templates():
    """Copy the HTML template to templates directory"""
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Copy the index.html content if it exists
    if os.path.exists('index.html'):
        with open('index.html', 'r') as src:
            with open(os.path.join('templates', 'index.html'), 'w') as dest:
                dest.write(src.read())
        print("Copied index.html to templates directory")
    else:
        print("Warning: index.html not found")

def setup():
    """Run the complete setup process"""
    print("Setting up LangGraph Chatbot...")
    setup_directories()
    install_requirements()
    copy_templates()
    print("\nSetup complete! You can now run the application with:")
    print("python app.py")

if __name__ == "__main__":
    setup()