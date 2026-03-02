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
    top_k = st.slider("Retrieval Depth (Top-K)", 1, 10, 5)

    st.info(
        f"**Target:** {FIXED_TICKER} (Apple Inc.)\n\n"
        "**Data:** FY2023-2024 10-K/Q Filings"
    )

    if st.button("🧹 Clear Analysis History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.streaming_done = False
        st.session_state.answer_saved = False
        st.session_state.last_answer = None
        st.session_state.last_sources = []
        st.rerun()

# ==========================================
# 3. HEADER
# ==========================================
st.title("🍎 Apple Financial Analyst")
st.markdown(
    f"""
This system provides high-precision answers based strictly on  
**Apple Inc. ({FIXED_TICKER})** SEC filings.
"""
)

# ==========================================
# 4. SESSION STATE INIT
# ==========================================
for key, default in {
    "messages": [],
    "streaming_done": False,
    "answer_saved": False,
    "last_answer": None,
    "last_sources": [],
    "last_provider": None,
    "last_cached": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ==========================================
# 5. RENDER CHAT HISTORY
# ==========================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 6. SAFE SCORE NORMALIZATION
# ==========================================
def normalize_score(raw_score):
    try:
        raw_score = float(raw_score)
    except:
        raw_score = 0.0

    if math.isnan(raw_score) or math.isinf(raw_score):
        raw_score = 0.0

    raw_score = max(min(raw_score, 10), -10)
    norm_score = 1 / (1 + math.exp(-raw_score))
    return max(0.0, min(1.0, norm_score))

# ==========================================
# 7. CHAT INPUT
# ==========================================
if prompt := st.chat_input("Ask about Apple's Q3 revenue, R&D growth, or risk factors..."):

    # Reset streaming state for new prompt
    st.session_state.streaming_done = False
    st.session_state.answer_saved = False
    st.session_state.last_answer = None
    st.session_state.last_sources = []

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # ======================
        # RUN ONLY ONCE
        # ======================
        if not st.session_state.streaming_done:

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
                        sources = data.get("sources", [])
                        provider = data.get("provider", "LLM")
                        is_cached = data.get("cached", False)

                        # Save for reruns
                        st.session_state.last_answer = answer
                        st.session_state.last_sources = sources
                        st.session_state.last_provider = provider
                        st.session_state.last_cached = is_cached

                        # Typing animation (once)
                        full_text = ""
                        for word in answer.split():
                            full_text += word + " "
                            message_placeholder.markdown(full_text + "▌")
                            time.sleep(0.008)

                        message_placeholder.markdown(full_text)

                        st.session_state.streaming_done = True

                    else:
                        st.error(f"Analysis failed. Backend returned: {response.status_code}")

                except Exception as e:
                    st.error(f"Connection Error: {e}")

        else:
            # Rerun path → show final instantly
            message_placeholder.markdown(st.session_state.last_answer or "")

    # ======================================
    # METADATA ROW (PERSISTENT)
    # ======================================
    if st.session_state.last_answer:
        cols = st.columns(3)

        with cols[0]:
            if st.session_state.last_cached:
                st.caption("⚡ **Source:** Semantic Cache (Postgres)")
            else:
                st.caption(f"🤖 **Inference:** {st.session_state.last_provider}")

        with cols[1]:
            st.caption(f"📊 **Chunks Retrieved:** {len(st.session_state.last_sources)}")

    # ======================================
    # SOURCE EVIDENCE
    # ======================================
    if st.session_state.last_sources:
        with st.expander("📚 View Document Evidence"):
            for i, src in enumerate(st.session_state.last_sources):
                score = normalize_score(src.get("score", 0.0))

                st.markdown(f"**Chunk {i+1}** | Relevancy: `{score:.2%}`")
                st.progress(score)
                st.caption(src.get("text", "")[:400] + "...")
                st.divider()

    # ======================================
    # SAVE ASSISTANT MESSAGE ONCE
    # ======================================
    if (
        st.session_state.streaming_done
        and not st.session_state.answer_saved
        and st.session_state.last_answer
    ):
        st.session_state.messages.append(
            {"role": "assistant", "content": st.session_state.last_answer}
        )
        st.session_state.answer_saved = True