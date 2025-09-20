from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)


# Use local SentenceTransformer for embeddings
class LocalEmbeddingFunction:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()
    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

embedding_function = LocalEmbeddingFunction()

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    try:
        print(f"[DEBUG] Loading and splitting document: {file_path}")
        splits = load_and_split_document(file_path)
        print(f"[DEBUG] Number of splits: {len(splits)}")
        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
        print(f"[DEBUG] Adding splits to ChromaDB for file_id: {file_id}")
        vectorstore.add_documents(splits)
        # vectorstore._collection.persist()
        print(f"[DEBUG] Document indexed and persisted for file_id: {file_id}")
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False

def delete_doc_from_chroma(file_id: int):
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
        
        vectorstore._collection.delete(where={"file_id": file_id})
        # vectorstore._collection.persist()
        print(f"Deleted all documents with file_id {file_id} and persisted changes.")
        
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False