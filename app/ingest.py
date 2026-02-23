from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

print("Loading PDF...")
loader = PyPDFLoader("./data/sample.pdf")
documents = loader.load()

print("Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

print("Creating embeddings...")
embedding = OllamaEmbeddings(model="nomic-embed-text")

print("Storing into Chroma...")
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db"
)

vectordb.persist()

print("Documents successfully embedded and stored!")