from flask import Flask, request, jsonify
import os
from utils import search_articles, fetch_article_content, concatenate_content, generate_answer_with_langchain

# Load environment variables from .env file (if required)
# You can load environment variables using dotenv if needed
# from dotenv import load_dotenv
# load_dotenv()

app = Flask(__name__)
@app.route('/query', methods=['POST'])
def query():
    """
    Handles the POST request to '/query'. Extracts the query from the request,
    processes it through the search, concatenate, and generate functions,
    and returns the generated answer.
    """
    # Get the data/query from the request
    data = request.get_json()
    query_text = data.get("query", "")

    if not query_text:
        return jsonify({"error": "Query not provided."})

    print("Received query:", query_text)
    
    try:
        # Step 1: Search and scrape articles based on the query
        print("Step 1: Searching articles")
        articles = search_articles(query_text)
        
        if not articles:
            return jsonify({"error": "No articles found."})

        # Step 2: Concatenate content from the scraped articles
        print("Step 2: Concatenating content")
        article_contents = []
        for article in articles:
            content = fetch_article_content(article['url'])
            article_contents.append(content)

        # Concatenate the article contents into one string
        concatenated_content = concatenate_content(articles)

        # Step 3: Generate an answer using the LLM (LangChain)
        print("Step 3: Generating answer")
        answer = generate_answer_with_langchain(concatenated_content, query_text)

        # Return the generated answer as a JSON response
        return jsonify({"answer": answer})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Run the Flask application on port 5001
    app.run(host='localhost', port=5001)
