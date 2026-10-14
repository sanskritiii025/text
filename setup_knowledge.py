from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

OLLAMA_MODEL = "phi3"

loader = TextLoader("./company_policies.txt")
docs = loader.load()

# Split into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

print(f"Split into {len(split_docs)} chunks. Building vector store...")

vector_db = Chroma.from_documents(
    documents=split_docs,
    embedding=OllamaEmbeddings(model=OLLAMA_MODEL),
    persist_directory="./sales_memory"
)
print(f" I have read your policy document and memorized it. ({len(split_docs)} chunks stored)")