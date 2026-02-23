import streamlit as st
import requests

# Set up the page
st.set_page_config(page_title="Financial RAG Assistant", page_icon="📈", layout="centered")
st.title("📈 Wall Street AI Analyst")
st.caption("Ask questions about Apple's (AAPL) latest SEC filings.")

# Define the FastAPI backend URL
API_URL = "http://localhost:8000/ask"

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for user input
if prompt := st.chat_input("E.g., What are the major supply chain risks?"):
    
    # 1. Display the user's message in the UI
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Send the question to our FastAPI backend
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Analyzing SEC Filings..."):
            try:
                # The payload matches the QueryRequest schema in main.py
                payload = {
                    "query": prompt,
                    "ticker": "AAPL",
                    "top_k": 5
                }
                
                # Make the POST request to FastAPI
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    # Extract our new advanced metadata
                    is_cached = data.get("cached", False)
                    provider = data.get("provider", "Groq")
                    
                    # Display the AI's answer
                    message_placeholder.markdown(answer)
                    
                    # --- NEW: Display the Engine Metadata ---
                    if is_cached:
                        st.caption("⚡ **Blazing Fast:** Retrieved directly from Postgres Semantic Cache")
                    else:
                        st.caption(f"🤖 **Engine:** Generated dynamically using {provider}")
                    
                    # Display the exact SEC sources in an expandable box
                    with st.expander("📚 View SEC Sources"):
                        for i, source in enumerate(sources):
                            st.markdown(f"**Source {i+1} (Relevance: {source['score']:.2f}):**")
                            st.caption(source['text'])
                            st.divider()
                            
                    # Add AI response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
            except Exception as e:
                st.error(f"Failed to connect to the backend. Is FastAPI running? Error: {e}")