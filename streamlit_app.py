import streamlit as st
import requests

# API URL for FastAPI backend
api_url = "http://127.0.0.1:8000/query"  # adjust if running on a different host/port

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
