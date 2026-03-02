import streamlit as st
import requests
import os
import math

# ==========================================
# 🌐 HARDCODED BACKEND CONFIG
# ==========================================
# Using your provided AWS Elastic IP directly
AWS_IP = "13.232.197.229"

# Prefer the Docker service name if available in environment, fallback to AWS IP
BACKEND_HOST = os.getenv("BACKEND_URL", f"http://{AWS_IP}:8001")
ASK_URL = f"{BACKEND_HOST}/ask"
HEALTH_URL = f"{BACKEND_HOST}/ready"

# ==========================================
# 🎨 CHATGPT DARK DESIGN (FORCED)
# ==========================================
st.set_page_config(
    page_title="Apple Financial Intelligence",
    page_icon="🍎",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0d0d0d !important; }
    .main .block-container {
        max-width: 850px;
        padding-top: 2rem;
    }
    .stApp, .stMarkdown, p, h1, h2, h3, h4, span, label, .stCaption {
        color: #ececec !important;
        font-family: 'Inter', sans-serif;
    }
    .stChatMessage { 
        background-color: transparent !important;
        border-radius: 0px; 
        padding: 1.5rem 0rem;
        border-bottom: 1px solid #2d2d2d;
    }
    [data-testid="stChatMessageAssistant"] {
        background-color: #1a1a1a !important;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 10px;
    }
    .stChatInputContainer textarea {
        background-color: #212121 !important;
        color: white !important;
        border: 1px solid #424242 !important;
        border-radius: 14px !important;
    }
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #2d2d2d;
    }
    .stProgress > div > div > div > div {
        background-color: #10a37f !important;
    }
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR & STATE
# ==========================================
FIXED_TICKER = "AAPL"

with st.sidebar:
    st.title("🍎 Apple Intelligence")
    # Simple check to see if we're hitting the AWS IP or a local/service name
    is_aws = AWS_IP in BACKEND_HOST
    st.caption(f"Endpoint: {BACKEND_HOST}")
    
    st.divider()

    try:
        # Check health of the backend
        r = requests.get(HEALTH_URL, timeout=5)
        if r.status_code == 200: st.success("● Core Engine: Online")
        else: st.warning("● Core Engine: Initializing")
    except Exception as e:
        st.error(f"● Core Engine: Offline")

    st.markdown("### 🛠️ Model Settings")
    top_k = st.slider("Retrieval Depth", 1, 10, 5)

    if st.button("🧹 Clear Conversation", use_container_width=True):
        for key in ["messages", "streaming_done", "answer_saved", "last_answer", "last_sources"]:
            if key in st.session_state: del st.session_state[key]
        st.rerun()

# Init Session State
for key, default in {
    "messages": [], "streaming_done": False, "answer_saved": False,
    "last_answer": None, "last_sources": [], "last_provider": None, "last_cached": False,
}.items():
    if key not in st.session_state: st.session_state[key] = default

# ==========================================
# 3. CHAT INTERFACE
# ==========================================
st.title("🍎 Apple Financial Analyst")

# Render History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def normalize_score(raw_score):
    try:
        raw_score = float(raw_score)
        # Sigmoid normalization for Reranker scores
        return 1 / (1 + math.exp(-max(min(raw_score, 10), -10)))
    except: return 0.0

# Input Logic
if prompt := st.chat_input("Ask about Apple's margins, iPhone sales, or net income..."):

    st.session_state.streaming_done = False
    st.session_state.answer_saved = False

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response = None
        with st.spinner(""):
            try:
                payload = {"query": prompt, "ticker": FIXED_TICKER, "top_k": top_k}
                response = requests.post(ASK_URL, json=payload, timeout=95)
            except Exception as e:
                st.error(f"Connection Error: {e}")

        if response and response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "").replace("$", r"\$")
            
            st.session_state.last_answer = answer
            st.session_state.last_sources = data.get("sources", [])
            st.session_state.last_provider = data.get("provider", "LLM")
            st.session_state.last_cached = data.get("cached", False)

            st.markdown(answer)
            st.session_state.streaming_done = True
        elif response:
            st.error(f"Backend Error ({response.status_code}): {response.text}")

    # Metadata & Evidence Row
    if st.session_state.last_answer:
        st.caption(f"🤖 **Model:** {st.session_state.last_provider} | {'⚡ Cached' if st.session_state.last_cached else '🌐 Live'}")
        
        if st.session_state.last_sources:
            with st.expander("📚 View SEC Filing Evidence"):
                for i, src in enumerate(st.session_state.last_sources):
                    score = normalize_score(src.get("score", 0.0))
                    st.markdown(f"**Source {i+1}** | Relevancy: `{score:.1%}`")
                    st.progress(score)
                    st.caption(src.get("text", "")[:400] + "...")
                    st.divider()

    # Save Assistant Message to history
    if st.session_state.streaming_done and not st.session_state.answer_saved:
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.last_answer})
        st.session_state.answer_saved = True