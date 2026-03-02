import streamlit as st
import requests
import os
import time
import math

# ==========================================
# 1. CONFIGURATION & THEMING
# ==========================================
st.set_page_config(
    page_title="Apple Financial Intelligence",
    page_icon="🍎",
    layout="wide"
)

FIXED_TICKER = "AAPL"
BACKEND_HOST = os.getenv("BACKEND_URL", "http://localhost:8001")
ASK_URL = f"{BACKEND_HOST}/ask"
FEEDBACK_URL = f"{BACKEND_HOST}/feedback"
HEALTH_URL = f"{BACKEND_HOST}/ready"

st.markdown("""
<style>
.stApp { background-color: #ffffff; }
.stChatMessage { border-radius: 15px; }
.stButton>button { border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🍎 Apple Intelligence")
    st.caption("Specialized Financial RAG Engine")

    st.divider()

    try:
        r = requests.get(HEALTH_URL, timeout=2)
        if r.status_code == 200:
            st.success("● Core Engine: Online")
        else:
            st.warning("● Core Engine: Initializing")
    except:
        st.error("● Core Engine: Offline")

    st.markdown("### 🛠️ Model Settings")
    top_k = st.slider(
        "Retrieval Depth (Top-K)", 1, 10, 5,
        help="Number of SEC filing chunks retrieved per query."
    )

    st.info(
        f"**Target:** {FIXED_TICKER} (Apple Inc.)\n\n"
        "**Data:** FY2023-2024 10-K/Q Filings"
    )

    st.divider()

    with st.expander("🚀 Tech Highlights"):
        st.write("- **Hybrid RAG:** Groq/Gemini LPU inference.")
        st.write("- **Vector DB:** Qdrant with Metadata Filtering.")
        st.write("- **Efficiency:** Semantic Caching in PostgreSQL.")
        st.write("- **Accuracy:** Cross-Encoder Reranking.")

    if st.button("🧹 Clear Analysis History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 3. HEADER
# ==========================================
st.title("🍎 Apple Financial Analyst")
st.markdown(
    f"""
This system provides high-precision answers based strictly on  
**Apple Inc. ({FIXED_TICKER})** SEC filings.

It leverages semantic search to bypass the noise of standard LLMs.
"""
)

# ==========================================
# 4. CHAT STATE
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 5. HELPER: SAFE SCORE NORMALIZATION
# ==========================================
def normalize_score(raw_score):
    try:
        raw_score = float(raw_score)
    except:
        raw_score = 0.0

    if math.isnan(raw_score) or math.isinf(raw_score):
        raw_score = 0.0

    # Prevent sigmoid overflow
    raw_score = max(min(raw_score, 10), -10)

    norm_score = 1 / (1 + math.exp(-raw_score))

    return max(0.0, min(1.0, norm_score))

# ==========================================
# 6. CHAT INPUT
# ==========================================
if prompt := st.chat_input(
    "Ask about Apple's Q3 revenue, R&D growth, or risk factors..."
):

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Retrieving SEC Filings & Reranking Context..."):
            try:
                payload = {
                    "query": prompt,
                    "ticker": FIXED_TICKER,
                    "top_k": top_k
                }

                response = requests.post(ASK_URL, json=payload, timeout=60)

                if response.status_code == 200:
                    data = response.json()

                    answer = data.get("answer", "No analysis available.")
                    is_cached = data.get("cached", False)
                    provider = data.get("provider", "LLM")
                    sources = data.get("sources", [])

                    # ======================
                    # Typing effect
                    # ======================
                    full_text = ""
                    for word in answer.split():
                        full_text += word + " "
                        message_placeholder.markdown(full_text + "▌")
                        time.sleep(0.008)
                    message_placeholder.markdown(full_text)

                    # ======================
                    # Metadata Row
                    # ======================
                    cols = st.columns(3)

                    with cols[0]:
                        if is_cached:
                            st.caption("⚡ **Source:** Semantic Cache (Postgres)")
                        else:
                            st.caption(f"🤖 **Inference:** {provider}")

                    with cols[1]:
                        st.caption(f"📊 **Chunks Retrieved:** {len(sources)}")

                    # Optional latency display if backend sends it later
                    with cols[2]:
                        latency = data.get("llm_ms")
                        if latency:
                            st.caption(f"⏱️ **LLM:** {latency:.0f} ms")

                    # ======================
                    # Source Evidence
                    # ======================
                    if sources:
                        with st.expander("📚 View Document Evidence"):
                            for i, src in enumerate(sources):
                                score = normalize_score(src.get("score", 0.0))

                                st.markdown(
                                    f"**Chunk {i+1}** | Relevancy: `{score:.2%}`"
                                )
                                st.progress(score)
                                st.caption(src.get("text", "")[:400] + "...")
                                st.divider()

                else:
                    st.error(
                        f"Analysis failed. Backend returned: {response.status_code}"
                    )

            except Exception as e:
                st.error(
                    f"Connection Error: Ensure your EC2 endpoint is accessible. ({e})"
                )

    if "answer" in locals():
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )