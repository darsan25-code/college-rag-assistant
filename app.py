import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

st.set_page_config(page_title="College AI", layout="wide")
st.title("🎓 College AI Assistant")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(persist_directory="db", embedding_function=embeddings)

llm = Ollama(model="mistral")   # free local AI

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

query = st.text_input("Ask anything from your notes")

if query:
    answer = qa.run(query)
    st.write("### 🤖 AI Answer")
    st.write(answer)
