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
BACKEND_HOST = os.getenv("BACKEND_URL", "http://13.232.197.229:8001")
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
# SAFE SCORE NORMALIZATION
# ==========================================
def normalize_score(raw_score):
    try:
        raw_score = float(raw_score)
    except:
        raw_score = 0.0

    if math.isnan(raw_score) or math.isinf(raw_score):
        raw_score = 0.0

    # Prevent math overflow by capping extreme values
    raw_score = max(min(raw_score, 10), -10)
    
    # Sigmoid function: turns logits (like 5.24) into percentages (like 0.99)
    norm_score = 1 / (1 + math.exp(-raw_score))
    return max(0.0, min(1.0, norm_score))

# ==========================================
# 7. STREAM GENERATOR (SIMPLIFIED)
# ==========================================
def stream_tokens(text):
    words = text.split(" ")
    for word in words:
        yield word + " "
        time.sleep(0.015)

# ==========================================
# 8. CHAT INPUT
# ==========================================
if prompt := st.chat_input("Ask about Apple's Q3 revenue, R&D growth..."):

    # Reset streaming state for new prompt
    st.session_state.streaming_done = False
    st.session_state.answer_saved = False
    st.session_state.last_answer = None
    st.session_state.last_sources = []

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        # ======================
        # RUN ONLY ONCE
        # ======================
        if not st.session_state.streaming_done:

            response = None
            with st.spinner("Retrieving SEC Filings & Reranking Context..."):
                try:
                    payload = {"query": prompt, "ticker": FIXED_TICKER, "top_k": top_k}
                    response = requests.post(ASK_URL, json=payload, timeout=60)
                except Exception as e:
                    st.error(f"Connection Error: {e}")

            # 🛑 EXECUTE OUTSIDE THE SPINNER TO PREVENT INFINITE LOOPING 🛑
            if response.status_code == 200:
                        data = response.json()

                        answer = data.get("answer", "No analysis available.")
                        
                        # 🚨 THE FIX: Stop Streamlit from turning money into green math equations
                        answer = answer.replace("$", r"\$")
                        
                        sources = data.get("sources", [])
                        provider = data.get("provider", "LLM")
                        is_cached = data.get("cached", False)

                        # SAVE STATE
                        st.session_state.last_answer = answer
                        st.session_state.last_sources = sources
                        st.session_state.last_provider = provider
                        st.session_state.last_cached = is_cached

                        # Stream text safely
                        st.write_stream(stream_tokens(answer))
                        st.session_state.streaming_done = True

            elif response:
                st.error(f"Analysis failed. Backend returned: {response.status_code}")

        else:
            # Rerun path → show final instantly
            st.markdown(st.session_state.last_answer or "")

    # ======================================
    # METADATA ROW (PERSISTENT)
    # ======================================
    if st.session_state.last_answer:
        cols = st.columns(3)

        with cols[0]:
            if st.session_state.last_cached:
                st.caption("⚡ **Source:** Semantic Cache")
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
                    
                    # 1. Get the raw cross-encoder score (e.g. 5.24)
                    raw_score = src.get("score", 0.0)
                    
                    # 2. Use the normalize function to squash it to 0.0 - 1.0
                    normalized_score = normalize_score(raw_score)
                    
                    # 3. Double-clamp it just to be 100% safe for Streamlit
                    safe_score = max(0.0, min(1.0, float(normalized_score)))

                    st.markdown(f"**Chunk {i+1}** | Relevancy: `{safe_score:.2%}`")
                    st.progress(safe_score) # Now guaranteed to be valid!
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