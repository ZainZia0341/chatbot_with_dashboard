from langsmith import Client
import os
from dotenv import load_dotenv
from pprint import pprint

# Load environment variables from .env file
load_dotenv()

# Set LangChain tracing and API keys
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

# Initialize the LangSmith Client
client = Client()

# Retrieve the project runs
project_runs = client.list_runs(project_name=LANGCHAIN_PROJECT)

# Iterate over the generator and extract specific token info
for run in project_runs:
    # Access attributes directly
    run_id = run.id
    session_id = run.session_id
    prompt_tokens = run.prompt_tokens
    completion_tokens = run.completion_tokens
    total_tokens = run.total_tokens

    # Print the extracted data
    print(f"Run ID: {run_id}")
    print(f"Session ID: {session_id}")
    print(f"Prompt Tokens: {prompt_tokens}")
    print(f"Completion Tokens: {completion_tokens}")
    print(f"Total Tokens: {total_tokens}")
    print("-" * 40)
