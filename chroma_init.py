import os
import chardet
import fitz  # PyMuPDF for PDF processing
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

PERSIST_DIR = './chroma_db'
vectorstore = None

file_document_ids = {}  # Dictionary to track file names and their associated document IDs

def initialize_chroma(splits=None):
    global vectorstore
    if splits:
        print("Initializing Chroma with new documents...")
        vectorstore = Chroma.from_documents(documents=splits, persist_directory=PERSIST_DIR, embedding=OpenAIEmbeddings())
    elif os.path.exists(PERSIST_DIR) and not splits:
        print("Loading Chroma from the existing database...")
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=OpenAIEmbeddings())
    else:
        raise ValueError("No documents to initialize and Chroma database does not exist.")
    return vectorstore


def extract_text_from_pdf(pdf_path):
    """Extract text from the PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text


def push_files_to_chroma(file_names, directory='./uploaded_files/'):
    global file_document_ids
    documents = []
    
    for file_name in file_names:
        file_path = os.path.join(file_name)

        if file_name.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)  # Handle PDF files
        else:
            # Detect encoding for non-PDF files
            with open(file_path, 'rb') as raw_file:
                result = chardet.detect(raw_file.read(10000))  # Read first 10,000 bytes
                encoding = result['encoding']
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()

        document = Document(page_content=text, metadata={"file_name": file_name})
        documents.append(document)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Initialize Chroma with new documents and store document IDs
    vectorstore = initialize_chroma(splits=splits)
    file_document_ids[file_name] = [doc.metadata.get('file_name') for doc in splits]
    return vectorstore


def save_uploaded_file(uploaded_file, directory='./uploaded_files/'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


def list_uploaded_files(directory='./uploaded_files/'):
    return os.listdir(directory) if os.path.exists(directory) else []


def delete_uploaded_file(file_name, directory='./uploaded_files/'):
    file_path = os.path.join(directory, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    delete_vectors_from_chroma(file_name)


def delete_vectors_from_chroma(file_name):
    global file_document_ids
    vectorstore = initialize_chroma()

    if file_name in file_document_ids:
        document_ids = file_document_ids[file_name]
        vectorstore.delete(ids=document_ids)  # Delete the vectors associated with the file
        del file_document_ids[file_name]
        print(f"Deleted vectors for {file_name}")
    else:
        print(f"No vectors found for {file_name}")
