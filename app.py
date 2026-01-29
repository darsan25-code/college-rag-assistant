from utils.pdf_loader import read_pdf
from utils.chunker import chunk_text
from utils.vector_store import VectorStore
from utils.llm import generate_answer

def main():
    pdf_path = "data/sample.pdf"
    text = read_pdf(pdf_path)

    chunks = chunk_text(text)

    vector_store = VectorStore()
    vector_store.build_index(chunks)

    question = input("Ask a question from your PDF: ")

    retrieved_chunks = vector_store.search(question)

    answer = generate_answer(retrieved_chunks, question)

    print("\n🤖 AI Answer:\n")
    print(answer)

if __name__ == "__main__":
    main()
