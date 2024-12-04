import streamlit as st
import requests
import os

# Set up Streamlit interface
st.title("RAG Chatbot")
st.write("Ask me anything!")

user_input = st.text_input("Your question:")

# API URL for FastAPI backend
api_url = "http://127.0.0.1:8000/query"

if st.button("Get Response"):
    if user_input:
        # Send request to FastAPI
        try:
            response = requests.post(api_url, json={"query": user_input})
            if response.status_code == 200:
                result = response.json()
                st.write("Chatbot Response:", result["response"])
            else:
                st.write("Error:", response.json()["detail"])
        except Exception as e:
            st.write("Connection error:", e)
    else:
        st.write("Please enter a question.")
