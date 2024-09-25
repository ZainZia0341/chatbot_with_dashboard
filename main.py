# app.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from chroma_init import initialize_chroma
import os
from dotenv import load_dotenv
load_dotenv()


os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# MongoDB connection settings
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "chatbot_db"
COLLECTION_NAME = "conversation"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Initialize vectorstore
vectorstore = initialize_chroma()

# Core LLM setup
llm = ChatOpenAI(model="gpt-4o-mini", api_key = OPENAI_API_KEY)

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
    retriever = vectorstore.as_retriever()

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
