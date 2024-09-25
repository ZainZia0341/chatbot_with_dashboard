import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
from mongodb import load_conversations
from langsmith import Client
import os

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Load environment variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# Initialize LangSmith Client
client = Client()

def get_conversation_stats():
    """Fetches and calculates statistics for the dashboard."""
    
    conversations = load_conversations()
    total_conversations = len(conversations)
    token_usage_per_conversation = {}
    sentiment_per_conversation = {}
    total_tokens_used = 0
    
    # Process each conversation
    for session_id, messages in conversations.items():
        token_count = sum([len(message['content'].split()) for message in messages])  # Simple word-based token estimation
        total_tokens_used += token_count
        token_usage_per_conversation[session_id] = token_count
        
        # Calculate sentiment for the conversation
        conversation_text = " ".join([message['content'] for message in messages])
        sentiment_scores = analyzer.polarity_scores(conversation_text)
        sentiment_per_conversation[session_id] = sentiment_scores
    
    return {
        'total_conversations': total_conversations,
        'token_usage_per_conversation': token_usage_per_conversation,
        'total_tokens_used': total_tokens_used,
        'sentiment_per_conversation': sentiment_per_conversation,
    }

def plot_token_usage(token_usage_per_conversation):
    """Creates a bar chart for token usage per conversation."""
    
    session_ids = list(token_usage_per_conversation.keys())
    token_counts = list(token_usage_per_conversation.values())
    
    data = pd.DataFrame({
        'Session ID': session_ids,
        'Tokens Used': token_counts
    })
    
    st.bar_chart(data.set_index('Session ID'))

def plot_sentiment(sentiment_per_conversation):
    """Creates a pie chart or bar chart for sentiment analysis."""
    
    session_ids = []
    sentiments = []
    
    for session_id, sentiment_scores in sentiment_per_conversation.items():
        session_ids.append(session_id)
        if sentiment_scores['compound'] >= 0.05:
            sentiments.append('Positive')
        elif sentiment_scores['compound'] <= -0.05:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')
    
    sentiment_data = pd.DataFrame({
        'Session ID': session_ids,
        'Sentiment': sentiments
    })
    
    st.bar_chart(sentiment_data.set_index('Session ID'))


def get_token_usage_from_langsmith():
    """Fetch token usage data from LangSmith API."""
    
    # Retrieve the project runs
    project_runs = client.list_runs(project_name=LANGCHAIN_PROJECT)

    # Collect data in a list
    token_data = []
    for run in project_runs:
        run_id = run.id
        session_id = run.session_id
        prompt_tokens = run.prompt_tokens
        completion_tokens = run.completion_tokens
        total_tokens = run.total_tokens

        # Store the token data
        token_data.append({
            "Run ID": run_id,
            "Session ID": session_id,
            "Prompt Tokens": prompt_tokens,
            "Completion Tokens": completion_tokens,
            "Total Tokens": total_tokens
        })

    return token_data
