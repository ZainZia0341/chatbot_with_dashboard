# Chatbot Dashboard with Streamlit

This repository contains a Chatbot that uses Streamlit for the frontend, LangChain for creating the retrieval-augmented generation (RAG) chain, MongoDB for storing conversation history, and LangSmith to trace the logs for Dashboard analytics.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Workflow](#workflow)
- [Environment Variables (.env) Configuration](#environment-variables-env-configuration)
- [How to Set Up the Project](#how-to-set-up-the-project)
- [How to Run the Project Locally](#how-to-run-the-project-locally)
- [Upload PDF/TXT Files for Knowledge Base](#upload-pdftxt-files-for-knowledge-base)

---

## Introduction

This project is a Chatbot that lets users interact with an AI-based assistant using RAG (Retrieval-Augmented Generation). The chatbot uses LangChain to manage conversation history, and MongoDB is used to store user conversations. A detailed analytics dashboard is included to visualize token usage and sentiment analysis for each conversation.

## Features
- **Used ChromaDB:** For RAG (Retrieval-Augmented Generation) chromadb vector database.
- **Chatbot Conversation:** Users can start new conversations, continue existing conversations, and get answers to their questions based on a retrieval-augmented generation system.
- **LangSmith** To trace the logs and display dashboard feature that shows the total token usage per conversation.
- **MongoDB Integration:** Conversation history is stored and retrieved from MongoDB.
- **Custom Knowledge Base:** Upload your own PDF or TXT files to create a custom knowledge base for the chatbot to use.
- **File Management:** Easily delete uploaded files anytime from the interface.
- **Multiple File Uploads Supported:** Users can upload multiple files for the knowledge base at once.

---

## Technologies Used

- **Streamlit**: Frontend UI framework.
- **LangChain**: Manages the RAG chain and message history.
- **LangSmith**: Trace logs for dashboard analytics.
- **MongoDB**: Stores conversation histories.
- **ChromaDB**: Retrieval-Augmented Generation.
- **Python**: The language used for developing the application.

---

## Workflow

1. **Conversation Flow:**
   - Users interact with the chatbot via the Streamlit app.
   - The question is passed to the RAG chain, which retrieves relevant documents and generates a concise answer.
   - The conversation is stored in MongoDB with the user's session ID.

2. **Dashboard Flow:**
   - Token usage is tracked for each conversation.
   - Sentiment analysis is done using VADER, and results are displayed in bar charts.
   - Both token usage and sentiment are visualized in a Streamlit dashboard.

3. **RAG Chain:**
   - A question is passed to LangChain’s retrieval-augmented generation (RAG) chain.
   - Context is retrieved from a vector store (Chroma).
   - LangChain processes the retrieved context and generates an answer using GPT-4.

4. **MongoDB Storage:**
   - Conversations are stored in MongoDB.
   - On starting a new conversation, a new session ID is generated.
   - Conversations can be retrieved by their session ID and displayed on the dashboard.

---

## Environment Variables (.env) Configuration

The `.env` file is essential for configuring your environment. You should create this file in the root directory of the project. Below is a sample of what it should contain:


MONGODB_URI=mongodb+srv://<username>:<password>@<cluster-url>/<database-name>?retryWrites=true&w=majority

OPENAI_API_KEY=<your_openai_api_key>

LANGCHAIN_PROJECT=<your_langchain_project_name>

## How to Set Up the Project

### Clone the repository:

git clone https://github.com/zainzia0341/Chatbot_Dashboard.git  
cd Chatbot_Dashboard

### Install the dependencies:

Ensure you have Python installed, preferably version 3.9.x, and then install the required dependencies using pip:

pip install -r requirements.txt

### Set up MongoDB:

1. Create a MongoDB Atlas account if you don’t have one.
2. Set up a database named `chatbot_db` and a collection named `conversation`.
3. Make sure to replace the `MONGODB_URI` in your `.env` file with your own MongoDB connection string.

---

## How to Run the Project Locally

### Set up environment variables:

Create a `.env` file in the project root directory with the values as shown in the [Environment Variables (.env) Configuration](#environment-variables-env-configuration) section.

### Use the following command to start the Streamlit app:

streamlit run app.py
