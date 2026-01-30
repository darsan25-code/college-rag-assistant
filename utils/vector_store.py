from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_PATH = r"C:\XboxGames\college-rag-assistant\data\UNIT III  notes.pdf"

print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print("Splitting text...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

print("Loading local embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating vector database...")
db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="db"
)

db.persist()
print("Vector database created successfully!")
