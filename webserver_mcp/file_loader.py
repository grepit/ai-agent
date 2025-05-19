# file_loader.py
import os
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

# Get the current directory (where file_loader.py is located)
current_dir = os.path.dirname(os.path.abspath(__file__))
files_dir = os.path.join(current_dir, "files_lib")
print(f"Current directory: {current_dir}")
print(f"Files directory: {files_dir}")

# Check if directory exists
if not os.path.exists(files_dir):
    os.makedirs(files_dir)
    print(f"Created directory: {files_dir}")

# Load all .txt or .md files from a folder
loader = DirectoryLoader(files_dir, glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

if not docs:
    print("Warning: No .txt files found in the files_lib directory")

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Create vector store using Ollama embeddings
embeddings = OllamaEmbeddings(model="mistral")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

