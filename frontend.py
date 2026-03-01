import streamlit as st
import requests
import os
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="📈",
    layout="centered"
)

BACKEND_HOST = os.getenv("BACKEND_URL", "http://localhost:8001")
ASK_URL = f"{BACKEND_HOST}/ask"
FEEDBACK_URL = f"{BACKEND_HOST}/feedback"
HEALTH_URL = f"{BACKEND_HOST}/ready"

st.title("📈 Wall Street AI Analyst")
st.caption("Advanced Financial RAG Engine powered by Groq & Gemini")

# ==========================================
# 2. BACKEND STATUS CHECK
# ==========================================
with st.sidebar:
    st.header("⚙️ System Status")

    try:
        r = requests.get(HEALTH_URL, timeout=3)
        if r.status_code == 200 and r.json().get("status") == "ready":
            st.success("Backend: Ready")
        else:
            st.warning("Backend: Not Ready")
    except:
        st.error("Backend: Offline")

    ticker = st.text_input("Ticker", value="AAPL").upper()
    top_k = st.slider("Top-K Chunks", 1, 10, 5)

    st.divider()
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 3. SESSION STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 4. DISPLAY CHAT HISTORY
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 5. USER INPUT
# ==========================================
if prompt := st.chat_input("E.g., Compare Apple's R&D spending to its revenue"):

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # ======================================
    # 6. CALL BACKEND
    # ======================================
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Analyzing SEC Filings & Cross-Referencing..."):
            try:
                payload = {
                    "query": prompt,
                    "ticker": ticker,
                    "top_k": top_k
                }

                response = requests.post(ASK_URL, json=payload, timeout=60)

                if response.status_code == 200:
                    data = response.json()

                    answer = data.get("answer", "No answer returned.")
                    query_hash = data.get("query_hash")
                    is_cached = data.get("cached", False)
                    provider = data.get("provider", "Unknown")
                    sources = data.get("sources", [])

                    # Simulated streaming effect
                    full_text = ""
                    for word in answer.split():
                        full_text += word + " "
                        message_placeholder.markdown(full_text)
                        time.sleep(0.01)

                    # ==================================
                    # METADATA DISPLAY
                    # ==================================
                    if is_cached:
                        st.caption("⚡ Semantic Cache Hit — served from PostgreSQL")
                    else:
                        st.caption(f"🤖 Live Inference — {provider}")

                    # ==================================
                    # SOURCES VIEWER
                    # ==================================
                    if sources:
                        with st.expander("📚 Source Context & Reranker Scores"):
                            for i, src in enumerate(sources):
                                st.markdown(
                                    f"**Source {i+1} | {src.get('document_type', 'SEC Filing')}**"
                                )
                                st.write(src.get("text", ""))

                                score = float(src.get("score", 0.0))
                                st.progress(score, text=f"Relevance Score: {score:.4f}")
                                st.divider()

                    # ==================================
                    # FEEDBACK
                    # ==================================
                    if query_hash:
                        st.write("---")
                        st.write("Rate this analysis:")

                        if f"voted_{query_hash}" not in st.session_state:
                            c1, c2, _ = st.columns([1, 1, 10])

                            with c1:
                                if st.button("👍", key=f"up_{query_hash}"):
                                    requests.post(
                                        FEEDBACK_URL,
                                        json={"query_hash": query_hash, "rating": 1},
                                        timeout=5,
                                    )
                                    st.session_state[f"voted_{query_hash}"] = True
                                    st.toast("Feedback recorded!")
                                    st.rerun()

                            with c2:
                                if st.button("👎", key=f"down_{query_hash}"):
                                    requests.post(
                                        FEEDBACK_URL,
                                        json={"query_hash": query_hash, "rating": -1},
                                        timeout=5,
                                    )
                                    st.session_state[f"voted_{query_hash}"] = True
                                    st.toast("Logged for review.")
                                    st.rerun()
                        else:
                            st.caption("Thank you for your feedback! ⭐")

                    # Save assistant message
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

                else:
                    st.error(f"Backend Error ({response.status_code}): {response.text}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Try reducing Top-K or check backend load.")

            except requests.exceptions.ConnectionError:
                st.error(
                    "🚨 Connection failed — ensure FastAPI backend is running and reachable."
                )

            except Exception as e:
                st.error(f"Unexpected error: {e}")