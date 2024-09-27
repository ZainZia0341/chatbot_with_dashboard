import os
from dotenv import load_dotenv
from chroma_init import  initialize_chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
import fitz  # PyMuPDF for PDF processing
from langchain.schema import Document

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
PERSIST_DIR = './chroma_db'

# MongoDB connection settings
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "chatbot_db"
COLLECTION_NAME = "conversation"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load default PDF file
DEFAULT_PDF_PATH = './default_document.pdf'

def load_pdf_content(pdf_path):
    """Load text content from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Load the default PDF without splitting since it's small
def process_default_pdf_to_splits():
    if os.path.exists(DEFAULT_PDF_PATH):
        print(f"Loading default PDF: {DEFAULT_PDF_PATH}")
        pdf_text = load_pdf_content(DEFAULT_PDF_PATH)
        # We don't split the text if the document is small
        return [Document(page_content=pdf_text, metadata={"file_name": "default_document.pdf"})]
    else:
        print("Default PDF not found. Skipping loading default PDF.")
        return None
    
# Process the default PDF for ChromaDB

# Initialize vectorstore with default PDF content if no user-uploaded files
default_splits = process_default_pdf_to_splits()

# Initialize vectorstore with default PDF content if no user-uploaded files
initialize_chroma(splits=default_splits)

# Core LLM setup
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

def get_session_history(session_id: str) -> MongoDBChatMessageHistory:
    """Retrieve the chat history stored in MongoDB"""
    return MongoDBChatMessageHistory(
        MONGODB_URI,
        session_id=session_id,
        database_name=DB_NAME,
        collection_name=COLLECTION_NAME
    )

def create_rag_chain():
    # Create retriever from Chroma
    retriever = initialize_chroma().as_retriever()

    # Define prompt templates
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question "
        "If you don't know the answer, say that you don't know."
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    retriever_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history,"
        "formulate a standalone question which can be understood without the chat history."
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
   
    # History-aware retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Final RAG chain with message history
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def invoke_rag_chain(question, session_id="default_session"):
    """Invoke the RAG chain with question and session ID for message history"""
    history = get_session_history(session_id)
    rag_chain = create_rag_chain()
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    # Invoke RAG chain with history
    result = conversational_rag_chain.invoke({"input": question}, config={"configurable": {"session_id": session_id}})
    return result["answer"], history.messages
