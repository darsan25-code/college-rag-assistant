import streamlit as st
import tempfile

from utils.pdf_loader import read_pdf
from utils.chunker import chunk_text
from utils.vector_store import VectorStore
from utils.llm import generate_answer

st.set_page_config(
    page_title="College Notes RAG Assistant",
    layout="centered"
)

st.title("📚 College Notes & Placement PDF AI Assistant")
st.caption("Ask questions based only on your uploaded PDFs")

# ---------- Session State ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat cleared!")

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown(
        "- Upload PDFs\n"
        "- Ask questions\n"
        "- AI Answers only from documents"
    )

# ---------- File Upload (Multiple PDFs) ----------
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    all_text = ""

    with st.spinner("Processing PDFs..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            text = read_pdf(pdf_path)
            all_text += text + "\n"

        chunks = chunk_text(all_text)

        vector_store = VectorStore()
        vector_store.build_index(chunks)

        st.session_state.vector_store = vector_store

    st.success(f"Processed {len(uploaded_files)} PDF(s) successfully!")

# ---------- Question Input ----------
question = st.text_input("Ask a question from your PDFs")

if question and st.session_state.vector_store:
    with st.spinner("Thinking..."):
        retrieved_chunks = st.session_state.vector_store.search(question)
        answer = generate_answer(retrieved_chunks, question)

    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "sources": retrieved_chunks
    })

# ---------- Chat Display ----------
if st.session_state.chat_history:
    st.divider()
    st.subheader("💬 Conversation")

    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {chat['question']}")
        st.markdown(f"**🤖 AI:** {chat['answer']}")

        with st.expander("📄 View source from PDF"):
            st.write(chat["sources"][0][:500])
