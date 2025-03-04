import streamlit as st
import requests
import uvicorn
import threading
import time

# 1. Import the FastAPI app from main.py
##from main import app as fastapi_app
##
### 2. Define a function to start Uvicorn
##def start_fastapi():
##    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="info")
##
### 3. Start the FastAPI server in a background thread if not started yet
##if "fastapi_started" not in st.session_state:
##    st.session_state["fastapi_started"] = True
##    thread = threading.Thread(target=start_fastapi, daemon=True)
##    thread.start()
##    # Give the server a moment to start
##    time.sleep(2)

# The rest of your Streamlit code remains the same:

# API URL for FastAPI backend
api_url = "http://127.0.0.1:8000/query"

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.title("RAG Chatbot")

# Create a user input box at the bottom of the screen (like ChatGPT)
user_input = st.chat_input("Ask me anything!")
if user_input:
    # 1. Add the user message to chat history
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    
    # 2. Send query to FastAPI and get the response
    try:
        response = requests.post(api_url, json={"query": user_input})
        if response.status_code == 200:
            result = response.json()
            # 3. Add the assistant's response to chat history
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": result.get("response", "")}
            )
        else:
            # If there's an error response from the API
            error_detail = response.json().get("detail")
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": f"Error from API: {error_detail}"}
            )
    except Exception as e:
        # If the request itself failed
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": f"Connection error: {str(e)}"}
        )

# Display the conversation in a ChatGPT-like format
for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
