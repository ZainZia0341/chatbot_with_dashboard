import streamlit as st
from main import invoke_rag_chain
from session_manager import generate_new_session_id
from mongodb import save_conversation, load_conversations, delete_conversation
from dashboard import display_dashboard
from chroma_init import push_files_to_chroma, list_uploaded_files, delete_uploaded_file, save_uploaded_file
import os

# Load environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state['page'] = 'conversation'

# Main layout - two columns: move the file uploader to the left side
col1, col2 = st.columns([2, 3])

# Inside col1: File uploader and conversation buttons (left side)
with col1:
    st.header("Upload Files for RAG")
    uploaded_files = st.file_uploader("Upload your .txt or .pdf files", type=['txt', 'pdf'], accept_multiple_files=True)

    if st.button("Push Files for RAG"):
        file_names = [save_uploaded_file(file) for file in uploaded_files]
        push_files_to_chroma(file_names)

    # Show uploaded files
    st.subheader("Uploaded Files:")
    uploaded_files_list = list_uploaded_files()
    if uploaded_files_list:
        for file_name in uploaded_files_list:
            if st.button(f"Delete {file_name}"):
                delete_uploaded_file(file_name)

    col3, col4 = st.columns(2)

    # Conversation Button
    with col3:
        if st.button("Conversation"):
            st.session_state['page'] = 'conversation'

    # Dashboard Button
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

# Inside col2: Main conversation/chatbot UI (on the right)
with col2:
    if st.session_state['page'] == 'conversation':
        st.header("Chat with the Assistant")
        user_input = st.text_input("Ask a question:")

        if st.button("Send"):
            if user_input:
                answer, full_chat_history = invoke_rag_chain(user_input, st.session_state['active_session_id'])
                st.session_state['conversations'][st.session_state['active_session_id']].append({"role": "User", "content": user_input})
                st.session_state['conversations'][st.session_state['active_session_id']].append({"role": "AI", "content": answer})
                save_conversation(st.session_state['active_session_id'], st.session_state['conversations'][st.session_state['active_session_id']])

        if st.session_state['active_session_id'] in st.session_state['conversations']:
            for message in st.session_state['conversations'][st.session_state['active_session_id']]:
                if message['role'] == "AI":
                    st.markdown(f"<div style='color:purple;'>Answer: {message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:green;'>User: {message['content']}</div>", unsafe_allow_html=True)

    elif st.session_state['page'] == 'dashboard':
        display_dashboard()  # Dashboard UI
