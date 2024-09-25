# mongodb.py

from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = "chatbot_db"
COLLECTION_NAME = "conversations"

# Initialize MongoDB client
client = MongoClient(MONGODB_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Function to save or update a conversation
def save_conversation(session_id, conversation):
    """Save or update a conversation in MongoDB."""
    collection.update_one(
        {"session_id": session_id},
        {"$set": {"conversation": conversation}},
        upsert=True
    )

# Function to load all conversations from MongoDB
def load_conversations():
    """Load all conversations from MongoDB."""
    conversations = {}
    for doc in collection.find():
        conversations[doc["session_id"]] = doc["conversation"]
    return conversations

# Function to delete a conversation from MongoDB
def delete_conversation(session_id):
    """Delete a conversation from MongoDB."""
    collection.delete_one({"session_id": session_id})

# Function to load a specific conversation by session_id
def load_conversation(session_id):
    """Load a specific conversation by session_id."""
    doc = collection.find_one({"session_id": session_id})
    if doc:
        return doc["conversation"]
    return []

# Function to get a list of all session IDs
def get_all_session_ids():
    """Retrieve all session IDs from MongoDB."""
    return [doc["session_id"] for doc in collection.find()]
