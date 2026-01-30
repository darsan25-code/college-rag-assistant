from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None

    def build_index(self, chunks):
        self.vector_store = FAISS.from_texts(chunks, self.embeddings)

    def search(self, query, k=3):
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
