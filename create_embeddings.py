import os
from langchain.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Directory with data
docs_path = "data"

# Load Documents
loader = DirectoryLoader(docs_path, glob="**/*.txt")
documents = loader.load()

# Strip documents for getting better results
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Here we choose the model, bga-small is good if you are using only cpu
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Crear el vectorstore FAISS
db = FAISS.from_documents(chunks, embeddings)

# Guardar el vectorstore para usarlo después (ruta relativa)
db_path = "knowledge_base.faiss"
db.save_local(db_path)

print(f"Embeddings creados y guardados en: {db_path}")
