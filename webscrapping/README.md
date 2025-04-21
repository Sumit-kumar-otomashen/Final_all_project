# Web Scraper Application

A Flask-based web application that allows users to scrape websites using LangChain and Groq.

## Features

- Web interface for easy URL input
- Scraping functionality using LangChain and Groq
- Modern and responsive UI
- Error handling and loading states

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd webscrapping
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with the following variables:
```
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_API_KEY=your_groq_api_key_here
FLASK_SECRET_KEY=your_secret_key_here
```

5. Run the application:
```bash
python run.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter the URL you want to scrape in the input field
2. Click the "Scrape" button
3. Wait for the results to appear
4. The results will be displayed in a formatted table below the input form

## Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure
- The application is set to run in debug mode by default. Change this in production

## License

MIT License 