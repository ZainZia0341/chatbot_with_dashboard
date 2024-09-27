# dashboard.py
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from mongodb import load_conversations
import os

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_conversation_stats():
    """Fetches and calculates statistics for the dashboard."""
    
    conversations = load_conversations()
    total_conversations = len(conversations)
    token_usage_per_conversation = {}
    sentiment_per_conversation = {}
    message_count_per_conversation = {}
    total_tokens_used = 0
    
    # Process each conversation
    for session_id, messages in conversations.items():
        token_count = sum([len(message['content'].split()) for message in messages])  # Simple word-based token estimation
        total_tokens_used += token_count
        token_usage_per_conversation[session_id] = token_count
        message_count_per_conversation[session_id] = len(messages)
        
        # Calculate sentiment for the conversation
        conversation_text = " ".join([message['content'] for message in messages])
        sentiment_scores = analyzer.polarity_scores(conversation_text)
        sentiment_per_conversation[session_id] = sentiment_scores
    
    return {
        'total_conversations': total_conversations,
        'token_usage_per_conversation': token_usage_per_conversation,
        'total_tokens_used': total_tokens_used,
        'sentiment_per_conversation': sentiment_per_conversation,
        'message_count_per_conversation': message_count_per_conversation
    }

def display_dashboard():
    """Displays the conversation statistics in a simple text format."""
    # Fetch conversation stats
    stats = get_conversation_stats()

    # Display total conversations and total tokens used
    st.header("Dashboard Analytics")
    st.write(f"Total Conversations: {stats['total_conversations']}")
    st.write(f"Total Tokens Used: {stats['total_tokens_used']}")

    # Display conversation details
    st.subheader("Conversation Details:")
    for session_id in stats['token_usage_per_conversation'].keys():
        st.write(f"**Session ID:** {session_id}")
        st.write(f"**Tokens Used:** {stats['token_usage_per_conversation'][session_id]}")
        st.write(f"**Sentiment:** {stats['sentiment_per_conversation'][session_id]['compound']}")
        st.write(f"**Number of Messages:** {stats['message_count_per_conversation'][session_id]}")
        st.write("---")
