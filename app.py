import streamlit as st
from openai import OpenAI
from tinydb import TinyDB
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
import tempfile
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="PRO Engineering Tutor",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# AI ANIMATED BACKGROUND (NO IMAGE)
# =====================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg,
        #0f2027,
        #203a43,
        #2c5364,
        #1f1c2c,
        #302b63
    );
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.block-container {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(18px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 0 40px rgba(0, 200, 255, 0.3);
}

h1, h2, h3, p, span, label {
    color: white !important;
}

textarea {
    background: rgba(0,0,0,0.45) !important;
    color: white !important;
    border-radius: 12px !important;
}

footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# TITLE
# =====================================================
st.title("🤖 PRO Engineering Tutor")
st.write("Memory + Multi-PDF Brain + Streaming + Voice AI")

# =====================================================
# OPENROUTER CLIENT
# =====================================================
client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# =====================================================
# MEMORY (TinyDB)
# =====================================================
db = TinyDB("memory.json")

if "messages" not in st.session_state:
    data = db.all()
    if data:
        st.session_state.messages = data[0]["messages"]
    else:
        st.session_state.messages = [
            {
                "role": "system",
                "content": "You are an expert engineering tutor. Explain clearly with examples."
            }
        ]

# =====================================================
# EMBEDDING MODEL
# =====================================================
if "embed_model" not in st.session_state:
    with st.spinner("🧠 Loading AI brain (first time ~20s)..."):
        st.session_state.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

embed_model = st.session_state.embed_model

# =====================================================
# WHISPER MODEL
# =====================================================
if "whisper_model" not in st.session_state:
    with st.spinner("🎤 Loading speech model..."):
        st.session_state.whisper_model = WhisperModel("base")

whisper_model = st.session_state.whisper_model

# =====================================================
# VECTOR DB
# =====================================================
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
    st.session_state.text_chunks = []

# =====================================================
# PDF UPLOAD
# =====================================================
uploaded_files = st.file_uploader(
    "📚 Upload Engineering PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    all_text = ""

    for file in uploaded_files:
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text

    chunks = [all_text[i:i+500] for i in range(0, len(all_text), 500)]
    embeddings = embed_model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    st.session_state.vector_db = index
    st.session_state.text_chunks = chunks

    st.success("✅ PDFs loaded. Ask questions from them!")

# =====================================================
# SHOW CHAT HISTORY
# =====================================================
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================================
# VOICE INPUT
# =====================================================
st.divider()
st.subheader("🎤 Voice Assistant")

audio = mic_recorder(start_prompt="🎤 Speak", stop_prompt="Stop")

voice_prompt = None

if audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio["bytes"])
        audio_path = f.name

    segments, _ = whisper_model.transcribe(audio_path)
    voice_prompt = "".join(segment.text for segment in segments)

    st.success(f"You said: {voice_prompt}")

# =====================================================
# TEXT INPUT
# =====================================================
text_prompt = st.chat_input("Ask anything about engineering...")
prompt = voice_prompt if voice_prompt else text_prompt

# =====================================================
# AI RESPONSE
# =====================================================
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    context = ""

    if st.session_state.vector_db:
        query_embedding = embed_model.encode([prompt])
        _, I = st.session_state.vector_db.search(np.array(query_embedding), k=3)
        context = "\n".join(st.session_state.text_chunks[i] for i in I[0])

        final_prompt = f"""
Use this PDF context when relevant:

{context}

Question:
{prompt}
"""
    else:
        final_prompt = prompt

    messages_for_api = st.session_state.messages + [
        {"role": "user", "content": final_prompt}
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_reply = ""

        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_for_api,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_reply += chunk.choices[0].delta.content
                placeholder.markdown(full_reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_reply}
    )

    db.truncate()
    db.insert({"messages": st.session_state.messages})

