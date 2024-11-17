import streamlit as st
import openai
import sys
import os

# Add the flask_app directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'flask_app'))

# Now you can import utils.py from flask_app
from utils import initialize_langchain, search_articles, fetch_article_content, concatenate_content, generate_answer_with_langchain

# Streamlit interface to input API keys
st.title("LLM-based RAG Search and Image Generation")
st.write("This app uses a Flask backend for querying, LangChain for generating responses, and OpenAI for image generation.")

# Input fields for API keys
OPENAI_API_KEY = st.text_input("Enter OpenAI API Key", type="password")
SERPER_API_KEY = st.text_input("Enter Serper API Key", type="password")

if OPENAI_API_KEY and SERPER_API_KEY:
    # Set OpenAI API key
    openai.api_key = OPENAI_API_KEY

    # Initialize LangChain with OpenAI API
    conversation_chain = initialize_langchain(OPENAI_API_KEY)

    # Input form for user's query
    query = st.text_input("Enter your query:")

    if query:
        # Search articles using the Serper API
        st.write("Searching for articles...")
        articles = search_articles(query, SERPER_API_KEY)

        if articles:
            # Concatenate the fetched articles into one text block
            content = concatenate_content(articles)

            # Generate an answer using LangChain
            st.write("Generating an answer...")
            answer = generate_answer_with_langchain(conversation_chain, content, query)
            st.markdown(f"<div style='font-family: Arial, sans-serif;'>{answer}</div>", unsafe_allow_html=True)

        else:
            st.warning("No articles found for the given query.")

        # Generate an image based on the same query
        st.write("Generating image based on your query...")

        try:
            # Make an API call to OpenAI to generate an image based on the query
            response = openai.Image.create(
                prompt=query,  # Use the same query as the image prompt
                n=1,  # Generate one image
                size="256x256"  # Set image size (adjust as needed)
            )

            # Extract the image URL from the API response
            image_url = response['data'][0]['url']

            # Display the generated image in the Streamlit app
            st.image(image_url, caption=f"Generated Image for: {query}", use_container_width=True)

        except Exception as e:
            st.error(f"Error generating image: {e}")

else:
    st.warning("Please enter valid API keys to continue.")
