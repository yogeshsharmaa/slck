import os
from flask import Flask
from slack_bolt import App
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
from datetime import datetime

# Langchain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from pathlib import Path
from unidecode import unidecode
import re

# Flask-SQLAlchemy for database
from flask_sqlalchemy import SQLAlchemy


load_dotenv()

app = Flask(__name__)
slack_token = os.getenv("SLACK_BOT_TOKEN")
slack_client = WebClient(token=slack_token)
signing_secret = os.getenv("SLACK_SIGNING_SECRET")
app_bolt = App(token=slack_token, signing_secret=signing_secret)

# Database configuration (SQLite)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat_history.db'
db = SQLAlchemy(app)

# Chat history model
class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50))
    message = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_bot = db.Column(db.Boolean, default=False)

# Global states for vector database and QA chain
vector_db_state = None
qa_chain_state = None

# Example usage of the API key (if needed)
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

# Initialize the document data and QA chain
def initialize():
    global vector_db_state, qa_chain_state

    # Load and process the PDF file
    if vector_db_state is None or qa_chain_state is None:
        pdf_files = [Path("testing_pdf.pdf")]
        chunk_size = 256
        chunk_overlap = 64

        # Create collection name
        collection_name = create_collection_name(pdf_files[0])

        # Load documents
        doc_splits = load_doc(pdf_files, chunk_size, chunk_overlap)

        # Create vector database
        vector_db = create_db(doc_splits, collection_name)
        vector_db_state = vector_db

        # Initialize QA chain
        llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
        temperature = 0.7
        max_tokens = 256
        top_k = 5
        qa_chain = initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db)
        qa_chain_state = qa_chain
    #... (your existing initialization functions for vector database and QA chain)

# Helper functions
def create_collection_name(filepath):
    collection_name = Path(filepath).stem
    collection_name = collection_name.replace(" ","-") 
    collection_name = unidecode(collection_name)
    collection_name = re.sub('[^A-Za-z0-9]+', '-', collection_name)
    collection_name = collection_name[:50]
    if len(collection_name) < 3:
        collection_name = collection_name + 'xyz'
    if not collection_name[0].isalnum():
        collection_name = 'A' + collection_name[1:]
    if not collection_name[-1].isalnum():
        collection_name = collection_name[:-1] + 'Z'
    return collection_name
    # ...(your existing helper functions)

def load_doc(list_file_path, chunk_size, chunk_overlap):
    loaders = [PyPDFLoader(str(x)) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

def create_db(splits, collection_name):
    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        collection_name=collection_name,
    )
    return vectordb
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db):
    llm = HuggingFaceEndpoint(
        repo_id=llm_model, 
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_k=top_k,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    retriever = vector_db.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain

def format_chat_history(chat_history):
    formatted_chat_history = []
    for message in chat_history:
        if message["type"] == "message":
            formatted_chat_history.append(f"User: {message['text']}")
    return formatted_chat_history

def conversation(message, history):
    formatted_chat_history = format_chat_history(history)
    result = qa_chain_state(
        {"question": message, "chat_history": formatted_chat_history}
    )
    response = result["answer"]
    return response



# Slack Events Handler

@app_bolt.event("app_mention")
def handle_app_mention(event, say):
    user_id = event["user"]
    user_message = event["text"]
    channel_id = event["channel"]

    # Store the user's message
    new_message = ChatHistory(user_id=user_id, message=user_message, is_bot=False)
    db.session.add(new_message)
    db.session.commit()

    # Retrieve history
    history = ChatHistory.query.filter_by(user_id=user_id).order_by(ChatHistory.timestamp).all()
    formatted_history = [{"type": "user" if not msg.is_bot else "bot", "text": msg.message} for msg in history]

    initialize()  # Initialize the LLM and VectorDB
    response_answer = conversation(user_message, formatted_history)

    # Store the bot's response
    new_message = ChatHistory(user_id=user_id, message=response_answer, is_bot=True)
    db.session.add(new_message)
    db.session.commit()

    say(text=response_answer, channel=channel_id)

# Slack Events Endpoint
@app.route('/slack/events', methods=['POST'])
def slack_events():
    if request.json.get("type") == "url_verification":
        # Handle the initial URL verification challenge from Slack
        challenge = request.json.get("challenge")
        return jsonify({"challenge": challenge})

    # Verify the request signature (rest of the code is the same as before)
    verifier = SignatureVerifier(signing_secret)
    if not verifier.is_valid_request(request.get_data(), request.headers):
        return jsonify({"error": "Invalid request signature"}), 401

    # Event handling logic starts here:
    event = request.json["event"]
    event_type = event.get("type")
    
     # Check if it's a message event
    if event_type == "app_mention":
        user_id = event["user"]
        user_message = event["text"]
        channel_id = event["channel"]

        # Store the user's message in the database
        new_message = ChatHistory(user_id=user_id, message=user_message, is_bot=False)
        db.session.add(new_message)
        db.session.commit()

        # Retrieve the conversation history from the database
        history = ChatHistory.query.filter_by(user_id=user_id).order_by(ChatHistory.timestamp).all()
        formatted_history = [{"type": "user" if not msg.is_bot else "bot", "text": msg.message} for msg in history]

        # Initialize the LLM and VectorDB
        initialize()
        response_answer = conversation(user_message, formatted_history)

        # Store the bot's response in the database
        new_message = ChatHistory(user_id=user_id, message=response_answer, is_bot=True)
        db.session.add(new_message)
        db.session.commit()

        # Send the bot's response back to Slack
        try:
            slack_client.chat_postMessage(channel=channel_id, text=response_answer)
        except SlackApiError as e:
            print(f"Error posting message: {e.response['error']}")

    return jsonify({"status": "ok"})

# Start the Flask server using Slack Bolt
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database tables if they don't exist
    port = int(os.environ.get("PORT", 5000))
    app_bolt.start(port=port)