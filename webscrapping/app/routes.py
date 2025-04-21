from flask import Blueprint, render_template, request, jsonify
from ..scrap import graph
import uuid

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/scrape', methods=['POST'])
def scrape():
    try:
        url = request.json.get('url')
        if not url:
            return jsonify({'error': 'URL is required'}), 400

        config = {"configurable": {"user_id": "12345", "thread_id": str(uuid.uuid4())}}
        initial_state = {
            "messages": [
                {"type": "human", "content": f"scrap this {url}, scrap this website and make table of top 2 movies with their ratings and year of release"}
            ]
        }
        
        result = graph.invoke(initial_state, config)
        return jsonify({'result': result['messages'][-1].content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500 