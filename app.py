import streamlit as st
from main import invoke_rag_chain
from session_manager import generate_new_session_id
from langchain_core.messages import AIMessage, HumanMessage
from mongodb import save_conversation, load_conversations, delete_conversation
from dashboard import get_conversation_stats, plot_token_usage, plot_sentiment, get_token_usage_from_langsmith
from langsmith import Client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set LangChain tracing and API keys
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state['page'] = 'conversation'

# Main layout - two columns: one narrow for conversation buttons, one wider for chatbot UI
col1, col2 = st.columns([2, 3])  # col1 (2 units) and col2 (3 units)

# Inside col1, create two equal buttons
with col1:
    col3, col4 = st.columns(2)  # Two equal columns inside col1

    # Button 1: Conversation
    with col3:
        if st.button("Conversation"):
            st.session_state['page'] = 'conversation'

    # Button 2: Dashboard
    with col4:
        if st.button("Dashboard"):
            st.session_state['page'] = 'dashboard'

    # Check if page is set to "conversation"
    if st.session_state['page'] == 'conversation':
        # Load conversations
        if 'conversations' not in st.session_state:
            st.session_state['conversations'] = load_conversations()

        if 'active_session_id' not in st.session_state:
            st.session_state['active_session_id'] = generate_new_session_id()
            st.session_state['conversations'][st.session_state['active_session_id']] = []

        # Start New Conversation Button
        if st.button("Start New Conversation"):
            new_session_id = generate_new_session_id()
            st.session_state['active_session_id'] = new_session_id
            st.session_state['conversations'][new_session_id] = []

        # Display conversations as buttons
        st.subheader("Your Conversations:")
        for session_id in st.session_state['conversations'].keys():
            if st.button(session_id):
                st.session_state['active_session_id'] = session_id

        # Display the active conversation ID
        st.write(f"Active Conversation ID: {st.session_state['active_session_id']}")

        # Delete Current Conversation button
        if st.button("Delete Current Conversation"):
            delete_conversation(st.session_state['active_session_id'])
            del st.session_state['conversations'][st.session_state['active_session_id']]
            if st.session_state['conversations']:
                st.session_state['active_session_id'] = list(st.session_state['conversations'].keys())[0]
            else:
                new_session_id = generate_new_session_id()
                st.session_state['active_session_id'] = new_session_id
                st.session_state['conversations'][new_session_id] = []

# Inside col2: Main conversation/chatbot UI
with col2:
    if st.session_state['page'] == 'conversation':
        # Chatbot UI
        st.header("Chat with the Assistant")

        # Input box for the user's question
        user_input = st.text_input("Ask a question:")
        if st.button("Send"):
            if user_input:
                # Invoke the RAG chain
                answer, full_chat_history = invoke_rag_chain(user_input, st.session_state['active_session_id'])
                
                # Save user's message and AI's response to the conversation history
                st.session_state['conversations'][st.session_state['active_session_id']].append({"role": "User", "content": user_input})
                st.session_state['conversations'][st.session_state['active_session_id']].append({"role": "AI", "content": answer})
                save_conversation(st.session_state['active_session_id'], st.session_state['conversations'][st.session_state['active_session_id']])

        # Display the conversation history
        if st.session_state['active_session_id'] in st.session_state['conversations']:
            for message in st.session_state['conversations'][st.session_state['active_session_id']]:
                if message['role'] == "AI":
                    st.markdown(f"<div style='color:purple;'>Answer: {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:green;'>User: {message['content']}</div>", unsafe_allow_html=True)

    elif st.session_state['page'] == 'dashboard':
        # Dashboard UI
        st.header("Dashboard Analytics")

        # Get conversation stats including token usage
        stats = get_conversation_stats()

        # Display total conversations and total tokens used
        st.write(f"Total Conversations: {stats['total_conversations']}")
        st.write(f"Total Tokens Used: {stats['total_tokens_used']}")

        # Display token usage per conversation
        st.subheader("Token Usage per Conversation")
        plot_token_usage(stats['token_usage_per_conversation'])

        # Display sentiment analysis per conversation
        st.subheader("Sentiment Analysis per Conversation")
        plot_sentiment(stats['sentiment_per_conversation'])

        # Display token usage from LangSmith
        st.subheader("Token Usage from LangSmith API")
        token_usage_data = get_token_usage_from_langsmith()
        st.write(token_usage_data)

