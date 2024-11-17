# utils.py

import http.client
import json
import requests
from bs4 import BeautifulSoup
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

# Function to initialize LangChain and set up the conversation chain
def initialize_langchain(openai_api_key):
    """
    Initializes LangChain with the provided OpenAI API key and sets up the conversation chain.
    """
    llm = OpenAI(api_key=openai_api_key)
    memory = ConversationBufferMemory()
    conversation_chain = ConversationChain(llm=llm, memory=memory)
    return conversation_chain

# Function to search articles using the Serper API
def search_articles(query, serper_api_key):
    """
    Searches for articles related to the query using the Serper API.
    Returns a list of dictionaries containing article URLs, headings, and snippets.
    """
    conn = http.client.HTTPSConnection("google.serper.dev")

    # Define the payload for the search query
    payload = json.dumps({
        "q": query  # This is the search query you are sending
    })

    # Set the request headers, including the API key and content type
    headers = {
        'X-API-KEY': serper_api_key,  # API key loaded from user input
        'Content-Type': 'application/json'
    }

    try:
        # Send the POST request to the /search endpoint with the payload and headers
        conn.request("POST", "/search", payload, headers)

        # Get the response from the API
        res = conn.getresponse()

        # Read and decode the response data
        data = res.read()

        # Decode the response data to JSON
        response_data = json.loads(data.decode("utf-8"))

        # Store the results in a list of dictionaries
        results = []
        for result in response_data.get('organic', []):
            result_dict = {
                "title": result.get('title', 'No Title Available'),
                "snippet": result.get('snippet', 'No Snippet Available'),
                "url": result.get('url', 'No URL Available')
            }
            results.append(result_dict)

        # Sort the results based on snippet length as an example
        sorted_results = sorted(results, key=lambda x: len(x['snippet']), reverse=True)

        # Get the top k results
        top_k_results = sorted_results[:5]

        return top_k_results
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Function to fetch article content using BeautifulSoup
def fetch_article_content(url):
    """
    Fetches the article content from the provided URL, extracting headings and text.
    Uses BeautifulSoup to parse the HTML.
    """
    content = ""
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract the main content from the article
            # This depends on the structure of the webpage, so you'll need to adjust accordingly
            paragraphs = soup.find_all('p')  # Extract all paragraph elements

            # Combine all paragraph text into a single string
            content = "\n".join([p.get_text() for p in paragraphs if p.get_text().strip()])

        else:
            content = "Failed to retrieve content"
    except Exception as e:
        content = f"An error occurred: {e}"

    return content.strip()

# Function to concatenate the content of the articles
def concatenate_content(articles):
    """
    Concatenates the content of the provided articles into a single string.
    """
    full_text = ""
    for article in articles:
        full_text += f"Title: {article['title']}\n"
        full_text += f"Snippet: {article['snippet']}\n"
        full_text += f"URL: {article['url']}\n"
        full_text += "\n"  # Add some space between articles

    return full_text

# Function to generate an answer using LangChain with the contextual content
def generate_answer_with_langchain(conversation_chain, content, query):
    """
    Generates an answer with LangChain, using conversation memory for multi-turn interaction.
    """
    try:
        # Update memory with current content and query
        conversation_chain.memory.chat_memory.add_user_message(content)
        conversation_chain.memory.chat_memory.add_user_message(query)

        # Generate a response using LangChain
        prompt = f"Based on the provided content, answer the question:\n\n{content}\n\nQuestion: {query}"
        response = conversation_chain.run(prompt)

        # Store response in memory for continuity
        conversation_chain.memory.chat_memory.add_ai_message(response)
        return response
    except Exception as e:
        print(f"Error in generate_answer_with_langchain: {e}")
        return "Error generating response."

